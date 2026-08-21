"""Tracking metrics with an explicit IoU association protocol."""

from __future__ import annotations

from collections import defaultdict

from roadsense.contracts import Detection, FrameRecord
from roadsense.geometry import greedy_iou_match


def _required_track_id(detection: Detection) -> int:
    if detection.track_id is None:
        raise ValueError("tracking detections require track_id")
    return detection.track_id


def evaluate_tracking(
    ground_truth: tuple[FrameRecord, ...],
    predictions: tuple[FrameRecord, ...],
    *,
    iou_threshold: float = 0.5,
) -> dict[str, object]:
    if len(ground_truth) != len(predictions) or not ground_truth:
        raise ValueError("ground-truth and prediction sequences must be non-empty and aligned")
    true_positives = false_positives = false_negatives = identity_switches = 0
    ground_truth_count = 0
    last_identity: dict[int, int] = {}
    association_counts: dict[tuple[int, int], int] = defaultdict(int)
    truth_track_lengths: dict[int, int] = defaultdict(int)
    prediction_track_lengths: dict[int, int] = defaultdict(int)
    for truth_frame, prediction_frame in zip(ground_truth, predictions, strict=True):
        if truth_frame.frame_index != prediction_frame.frame_index:
            raise ValueError("frame indices must align")
        if any(detection.track_id is None for detection in truth_frame.detections):
            raise ValueError("ground-truth tracking detections require track_id")
        if any(detection.track_id is None for detection in prediction_frame.detections):
            raise ValueError("predicted tracking detections require track_id")
        ground_truth_count += len(truth_frame.detections)
        for detection in truth_frame.detections:
            truth_track_lengths[_required_track_id(detection)] += 1
        for detection in prediction_frame.detections:
            prediction_track_lengths[_required_track_id(detection)] += 1
        result = greedy_iou_match(
            truth_frame.detections,
            prediction_frame.detections,
            iou_threshold=iou_threshold,
            class_aware=True,
        )
        true_positives += len(result.matches)
        false_negatives += len(result.unmatched_left)
        false_positives += len(result.unmatched_right)
        for match in result.matches:
            truth_id = _required_track_id(truth_frame.detections[match.left_index])
            prediction_id = _required_track_id(prediction_frame.detections[match.right_index])
            previous = last_identity.get(truth_id)
            if previous is not None and previous != prediction_id:
                identity_switches += 1
            last_identity[truth_id] = prediction_id
            association_counts[(truth_id, prediction_id)] += 1
    selected_pairs: set[tuple[int, int]] = set()
    used_truth: set[int] = set()
    used_prediction: set[int] = set()
    for pair, count in sorted(
        association_counts.items(), key=lambda item: (-item[1], item[0][0], item[0][1])
    ):
        truth_id, prediction_id = pair
        if truth_id in used_truth or prediction_id in used_prediction:
            continue
        selected_pairs.add(pair)
        used_truth.add(truth_id)
        used_prediction.add(prediction_id)
    identity_true_positive = sum(association_counts[pair] for pair in selected_pairs)
    total_truth_identity = sum(truth_track_lengths.values())
    total_prediction_identity = sum(prediction_track_lengths.values())
    identity_false_negative = total_truth_identity - identity_true_positive
    identity_false_positive = total_prediction_identity - identity_true_positive
    identity_f1 = (
        2
        * identity_true_positive
        / max(1, 2 * identity_true_positive + identity_false_positive + identity_false_negative)
    )
    mota = 1.0 - (false_negatives + false_positives + identity_switches) / max(
        1, ground_truth_count
    )
    return {
        "protocol": "roadsense.tracking-iou/v1",
        "iou_threshold": iou_threshold,
        "mota": mota,
        "identity_f1": identity_f1,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "identity_switches": identity_switches,
        "ground_truth_detections": ground_truth_count,
        "claim_boundary": (
            "identity_f1 uses deterministic maximum-count greedy identity assignment; "
            "it is not the TrackEval reference implementation."
        ),
    }
