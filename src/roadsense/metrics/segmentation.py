"""Semantic-segmentation confusion matrix and IoU metrics."""

from __future__ import annotations

import operator
from typing import SupportsIndex, cast

import numpy as np
from numpy.typing import NDArray


def _integer_parameter(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    try:
        return operator.index(cast(SupportsIndex, value))
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc


def evaluate_segmentation(
    ground_truth: NDArray[np.integer],
    predictions: NDArray[np.integer],
    *,
    num_classes: int,
    ignore_index: int = 255,
) -> dict[str, object]:
    num_classes = _integer_parameter(num_classes, "num_classes")
    ignore_index = _integer_parameter(ignore_index, "ignore_index")
    ground_truth_array = np.asarray(ground_truth)
    prediction_array = np.asarray(predictions)
    if ground_truth_array.shape != prediction_array.shape or ground_truth_array.size == 0:
        raise ValueError("segmentation arrays must be non-empty and share a shape")
    if not 2 <= num_classes <= 10_000:
        raise ValueError("num_classes must be in [2, 10000]")
    # The protocol returns a dense matrix.  Bound its size before allocating
    # ``num_classes²`` bins so an untrusted manifest cannot trigger a huge
    # memory allocation (4M int64 cells is approximately 32 MiB).
    if num_classes * num_classes > 4_000_000:
        raise ValueError("num_classes is too large for the dense confusion matrix")
    if not np.issubdtype(ground_truth_array.dtype, np.integer):
        raise ValueError("segmentation labels must use an integer dtype")
    truth_raw = ground_truth_array.reshape(-1)
    prediction_raw = prediction_array.reshape(-1)
    valid = truth_raw != ignore_index
    truth_raw = truth_raw[valid]
    prediction_raw = prediction_raw[valid]
    if truth_raw.size == 0:
        raise ValueError("all segmentation pixels are ignored")
    if not np.issubdtype(prediction_array.dtype, np.integer):
        raise ValueError("segmentation labels must use an integer dtype")
    # Avoid silent uint64 -> int64 wraparound before range checks.
    if np.issubdtype(truth_raw.dtype, np.unsignedinteger) and np.any(
        truth_raw > np.iinfo(np.int64).max
    ):
        raise ValueError("segmentation labels are outside the configured class range")
    if np.issubdtype(prediction_raw.dtype, np.unsignedinteger) and np.any(
        prediction_raw > np.iinfo(np.int64).max
    ):
        raise ValueError("segmentation labels are outside the configured class range")
    truth = truth_raw.astype(np.int64, copy=False)
    prediction = prediction_raw.astype(np.int64, copy=False)
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
