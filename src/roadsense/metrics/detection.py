"""Small, explicit detection metric protocol used by fixtures and adapter tests."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from numpy.typing import NDArray

from roadsense.contracts import Detection, FrameRecord
from roadsense.geometry import greedy_iou_match, validate_iou_threshold
from roadsense.metrics._validation import validate_aligned_sequences


def _average_precision(recalls: NDArray[np.float64], precisions: NDArray[np.float64]) -> float:
    recall_points = np.concatenate(([0.0], recalls, [1.0]))
    precision_points = np.concatenate(([0.0], precisions, [0.0]))
    for index in range(precision_points.size - 2, -1, -1):
        precision_points[index] = max(precision_points[index], precision_points[index + 1])
    changes = np.flatnonzero(recall_points[1:] != recall_points[:-1])
    return float(
        np.sum(
            (recall_points[changes + 1] - recall_points[changes]) * precision_points[changes + 1]
        )
    )


def evaluate_detection(
    ground_truth: tuple[FrameRecord, ...],
    predictions: tuple[FrameRecord, ...],
    *,
    iou_threshold: float = 0.5,
) -> dict[str, object]:
    validate_aligned_sequences(ground_truth, predictions)
    iou_threshold = validate_iou_threshold(iou_threshold)
    categories = sorted(
        {
            detection.category_id
            for frame in ground_truth + predictions
            for detection in frame.detections
        }
    )
    per_class: dict[str, dict[str, float | int]] = {}
    aggregate_tp = aggregate_fp = aggregate_fn = 0
    aps: list[float] = []
    for category_id in categories:
        truth_by_frame: dict[int, tuple[Detection, ...]] = {}
        truth_count = 0
        scored_predictions: list[tuple[float, int, float, float, float, float, int, Detection]] = []
        for truth_frame, prediction_frame in zip(ground_truth, predictions, strict=True):
            class_truth = tuple(
                sorted(
                    (
                        detection
                        for detection in truth_frame.detections
                        if detection.category_id == category_id
                    ),
                    key=lambda detection: (
                        detection.bbox.x_min,
                        detection.bbox.y_min,
                        detection.bbox.x_max,
                        detection.bbox.y_max,
                    ),
                )
            )
            truth_by_frame[truth_frame.frame_index] = class_truth
            truth_count += len(class_truth)
            scored_predictions.extend(
                (
                    detection.score,
                    prediction_frame.frame_index,
                    detection.bbox.x_min,
                    detection.bbox.y_min,
                    detection.bbox.x_max,
                    detection.bbox.y_max,
                    detection_index,
                    detection,
                )
                for detection_index, detection in enumerate(prediction_frame.detections)
                if detection.category_id == category_id
            )
        # AP is sensitive to the order of equal-score predictions.  Include
        # every geometric coordinate in the tie-break so a report is stable
        # even when the upstream adapter changes object iteration order.
        scored_predictions.sort(key=lambda item: (-item[0], *item[1:7]))
        claimed: dict[int, set[int]] = defaultdict(set)
        true_flags: list[float] = []
        false_flags: list[float] = []
        for (
            _score,
            frame_index,
            _x_min,
            _y_min,
            _x_max,
            _y_max,
            _detection_index,
            prediction,
        ) in scored_predictions:
            candidates = tuple(
                detection
                for index, detection in enumerate(truth_by_frame[frame_index])
                if index not in claimed[frame_index]
            )
            available_indices = tuple(
                index
                for index in range(len(truth_by_frame[frame_index]))
                if index not in claimed[frame_index]
            )
            result = greedy_iou_match((prediction,), candidates, iou_threshold=iou_threshold)
            if result.matches:
                claimed[frame_index].add(available_indices[result.matches[0].right_index])
                true_flags.append(1.0)
                false_flags.append(0.0)
            else:
                true_flags.append(0.0)
                false_flags.append(1.0)
        cumulative_tp = np.cumsum(np.asarray(true_flags, dtype=np.float64))
        cumulative_fp = np.cumsum(np.asarray(false_flags, dtype=np.float64))
        recalls = cumulative_tp / max(1, truth_count)
        precisions = cumulative_tp / np.maximum(1.0, cumulative_tp + cumulative_fp)
        ap = _average_precision(recalls, precisions) if truth_count else 0.0
        tp = int(cumulative_tp[-1]) if cumulative_tp.size else 0
        fp = int(cumulative_fp[-1]) if cumulative_fp.size else 0
        fn = truth_count - tp
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, truth_count)
        per_class[str(category_id)] = {
            "ap": ap,
            "precision": precision,
            "recall": recall,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "ground_truth_count": truth_count,
        }
        if truth_count:
            aps.append(ap)
        aggregate_tp += tp
        aggregate_fp += fp
        aggregate_fn += fn
    return {
        "protocol": "roadsense.detection-ap/v1",
        "iou_threshold": iou_threshold,
        "ap": float(np.mean(aps)) if aps else 0.0,
        "precision": aggregate_tp / max(1, aggregate_tp + aggregate_fp),
        "recall": aggregate_tp / max(1, aggregate_tp + aggregate_fn),
        "true_positives": aggregate_tp,
        "false_positives": aggregate_fp,
        "false_negatives": aggregate_fn,
        "per_class": per_class,
        "claim_boundary": "This compact protocol is not COCO mAP and must not be labeled as such.",
    }
