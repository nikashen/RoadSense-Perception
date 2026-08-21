from __future__ import annotations

import numpy as np
import pytest

from roadsense.metrics import evaluate_segmentation


def test_perfect_segmentation() -> None:
    truth = np.asarray([[0, 1], [1, 0]], dtype=np.int64)
    report = evaluate_segmentation(truth, truth, num_classes=2)
    assert report["pixel_accuracy"] == 1.0
    assert report["mean_iou"] == 1.0


def test_segmentation_confusion_and_iou() -> None:
    truth = np.asarray([[0, 0], [1, 1]], dtype=np.int64)
    prediction = np.asarray([[0, 1], [1, 1]], dtype=np.int64)
    report = evaluate_segmentation(truth, prediction, num_classes=2)
    assert report["confusion_matrix"] == [[1, 1], [0, 2]]
    assert report["pixel_accuracy"] == 0.75
    assert report["mean_iou"] == pytest.approx((0.5 + 2 / 3) / 2)


def test_ignore_index_is_excluded() -> None:
    truth = np.asarray([[0, 255], [1, 1]], dtype=np.int64)
    prediction = np.asarray([[0, 999], [1, 1]], dtype=np.int64)
    report = evaluate_segmentation(truth, prediction, num_classes=2)
    assert report["evaluated_pixels"] == 3
    assert report["pixel_accuracy"] == 1.0


def test_segmentation_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shape"):
        evaluate_segmentation(np.zeros((2, 2)), np.zeros((3, 2)), num_classes=2)


def test_segmentation_rejects_out_of_range_labels() -> None:
    with pytest.raises(ValueError, match="class range"):
        evaluate_segmentation(
            np.asarray([[0, 2]], dtype=np.int64),
            np.asarray([[0, 1]], dtype=np.int64),
            num_classes=2,
        )


def test_segmentation_rejects_all_ignored() -> None:
    with pytest.raises(ValueError, match="ignored"):
        evaluate_segmentation(
            np.full((2, 2), 255),
            np.zeros((2, 2)),
            num_classes=2,
        )
