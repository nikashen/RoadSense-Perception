from __future__ import annotations

import pytest

from roadsense.contracts import BoxXYXY, Detection, FrameRecord, ImageSize
from roadsense.metrics import evaluate_detection

SIZE = ImageSize(width=100, height=100)


def _frame(index: int, detections: tuple[Detection, ...]) -> FrameRecord:
    return FrameRecord(
        frame_index=index,
        timestamp_ms=index * 100,
        image_size=SIZE,
        detections=detections,
    )


def _detection(x: float, *, score: float = 1.0, category_id: int = 1) -> Detection:
    return Detection(
        category_id=category_id,
        score=score,
        bbox=BoxXYXY(x_min=x, y_min=10, x_max=x + 20, y_max=30),
    )


def test_perfect_detection_metrics() -> None:
    truth = (_frame(0, (_detection(10),)),)
    report = evaluate_detection(truth, truth)
    assert report["ap"] == 1.0
    assert report["precision"] == 1.0
    assert report["recall"] == 1.0


def test_false_positive_reduces_precision() -> None:
    truth = (_frame(0, (_detection(10),)),)
    predictions = (_frame(0, (_detection(10, score=0.9), _detection(60, score=0.8))),)
    report = evaluate_detection(truth, predictions)
    assert report["true_positives"] == 1
    assert report["false_positives"] == 1
    assert report["precision"] == 0.5


def test_missed_detection_reduces_recall() -> None:
    truth = (_frame(0, (_detection(10), _detection(60))),)
    predictions = (_frame(0, (_detection(10),)),)
    report = evaluate_detection(truth, predictions)
    assert report["recall"] == 0.5
    assert report["false_negatives"] == 1


def test_wrong_class_does_not_match() -> None:
    truth = (_frame(0, (_detection(10, category_id=1),)),)
    predictions = (_frame(0, (_detection(10, category_id=2),)),)
    report = evaluate_detection(truth, predictions)
    assert report["true_positives"] == 0


def test_detection_protocol_does_not_claim_coco_map() -> None:
    report = evaluate_detection((_frame(0, ()),), (_frame(0, ()),))
    assert report["protocol"] == "roadsense.detection-ap/v1"
    assert "not COCO mAP" in str(report["claim_boundary"])


def test_detection_requires_aligned_frames() -> None:
    with pytest.raises(ValueError, match="align"):
        evaluate_detection((_frame(0, ()),), (_frame(1, ()),))
