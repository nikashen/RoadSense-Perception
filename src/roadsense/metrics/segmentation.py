"""Semantic-segmentation confusion matrix and IoU metrics."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def evaluate_segmentation(
    ground_truth: NDArray[np.integer],
    predictions: NDArray[np.integer],
    *,
    num_classes: int,
    ignore_index: int = 255,
) -> dict[str, object]:
    if ground_truth.shape != predictions.shape or ground_truth.size == 0:
        raise ValueError("segmentation arrays must be non-empty and share a shape")
    if not 2 <= num_classes <= 10_000:
        raise ValueError("num_classes must be in [2, 10000]")
    truth = np.asarray(ground_truth, dtype=np.int64).reshape(-1)
    prediction = np.asarray(predictions, dtype=np.int64).reshape(-1)
    valid = truth != ignore_index
    truth = truth[valid]
    prediction = prediction[valid]
    if truth.size == 0:
        raise ValueError("all segmentation pixels are ignored")
    if (
        np.any(truth < 0)
        or np.any(truth >= num_classes)
        or np.any(prediction < 0)
        or np.any(prediction >= num_classes)
    ):
        raise ValueError("segmentation labels are outside the configured class range")
    encoded = truth * num_classes + prediction
    confusion = np.bincount(encoded, minlength=num_classes**2).reshape(num_classes, num_classes)
    true_positive = np.diag(confusion).astype(np.float64)
    truth_count = confusion.sum(axis=1, dtype=np.float64)
    prediction_count = confusion.sum(axis=0, dtype=np.float64)
    union = truth_count + prediction_count - true_positive
    present = union > 0
    iou = np.divide(
        true_positive,
        union,
        out=np.full(num_classes, np.nan, dtype=np.float64),
        where=present,
    )
    return {
        "protocol": "roadsense.semantic-iou/v1",
        "pixel_accuracy": float(true_positive.sum() / max(1.0, confusion.sum())),
        "mean_iou": float(np.nanmean(iou)),
        "per_class_iou": [None if np.isnan(value) else float(value) for value in iou],
        "confusion_matrix": confusion.tolist(),
        "evaluated_pixels": int(truth.size),
    }
