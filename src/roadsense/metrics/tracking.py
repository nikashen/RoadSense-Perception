"""Tracking metrics with an explicit IoU association protocol."""

from __future__ import annotations

from collections import defaultdict

from roadsense.contracts import Detection, FrameRecord
from roadsense.geometry import greedy_iou_match, validate_iou_threshold
from roadsense.metrics._validation import validate_aligned_sequences


def _required_track_id(detection: Detection) -> int:
    if detection.track_id is None:
        raise ValueError("tracking detections require track_id")
    return detection.track_id


def _record_track_category(
    detection: Detection,
    categories: dict[int, int],
    *,
    role: str,
) -> None:
    """Reject reusing one sequence track ID for multiple semantic classes."""

    track_id = _required_track_id(detection)
    previous_category = categories.get(track_id)
    if previous_category is not None and previous_category != detection.category_id:
        raise ValueError(
            f"{role} track_id {track_id} changes category across frames "
            f"({previous_category} -> {detection.category_id})"
        )
    categories[track_id] = detection.category_id


def evaluate_tracking(
    ground_truth: tuple[FrameRecord, ...],
    predictions: tuple[FrameRecord, ...],
    *,
    iou_threshold: float = 0.5,
) -> dict[str, object]:
    validate_aligned_sequences(ground_truth, predictions)
    iou_threshold = validate_iou_threshold(iou_threshold)
    true_positives = false_positives = false_negatives = identity_switches = 0
    ground_truth_count = 0
    last_identity: dict[int, int] = {}
    association_counts: dict[tuple[int, int], int] = defaultdict(int)
    truth_track_lengths: dict[int, int] = defaultdict(int)
    prediction_track_lengths: dict[int, int] = defaultdict(int)
    truth_track_categories: dict[int, int] = {}
    prediction_track_categories: dict[int, int] = {}
    for truth_frame, prediction_frame in zip(ground_truth, predictions, strict=True):
        if any(detection.track_id is None for detection in truth_frame.detections):
            raise ValueError("ground-truth tracking detections require track_id")
        if any(detection.track_id is None for detection in prediction_frame.detections):
            raise ValueError("predicted tracking detections require track_id")
        truth_ids = [detection.track_id for detection in truth_frame.detections]
        prediction_ids = [detection.track_id for detection in prediction_frame.detections]
        if len(truth_ids) != len(set(truth_ids)):
            raise ValueError("ground-truth track_id values must be unique within a frame")
        if len(prediction_ids) != len(set(prediction_ids)):
            raise ValueError("predicted track_id values must be unique within a frame")
        ground_truth_count += len(truth_frame.detections)
        for detection in truth_frame.detections:
            _record_track_category(detection, truth_track_categories, role="ground-truth")
            truth_track_lengths[_required_track_id(detection)] += 1
        for detection in prediction_frame.detections:
            _record_track_category(detection, prediction_track_categories, role="prediction")
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
    if ground_truth_count == 0:
        raise ValueError("tracking ground truth must contain at least one detection")
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
