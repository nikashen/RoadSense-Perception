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


def test_detection_rejects_duplicate_or_regressing_frame_sequence() -> None:
    frame_zero = _frame(0, ())
    with pytest.raises(ValueError, match="unique"):
        evaluate_detection((frame_zero, frame_zero), (frame_zero, frame_zero))

    frame_one = _frame(1, ())
    frame_two_with_regressed_timestamp = FrameRecord(
        frame_index=2,
        timestamp_ms=50,
        image_size=SIZE,
        detections=(),
    )
    with pytest.raises(ValueError, match="monotonic"):
        evaluate_detection(
            (frame_one, frame_two_with_regressed_timestamp),
            (frame_one, frame_two_with_regressed_timestamp),
        )


def test_detection_rejects_invalid_threshold_even_for_empty_frames() -> None:
    frame = _frame(0, ())
    with pytest.raises(ValueError, match="iou_threshold"):
        evaluate_detection((frame,), (frame,), iou_threshold=1.1)


def test_detection_is_invariant_to_ground_truth_detection_order() -> None:
    size = ImageSize(width=20, height=20)

    def frame(detections: tuple[Detection, ...]) -> FrameRecord:
        return FrameRecord(
            frame_index=0,
            timestamp_ms=0,
            image_size=size,
            detections=detections,
        )

    def box(x_min: float, x_max: float) -> Detection:
        return Detection(
            category_id=1,
            bbox=BoxXYXY(x_min=x_min, y_min=1, x_max=x_max, y_max=2),
        )

    truth = (box(3, 8), box(1, 6))
    predictions = (box(6, 9), box(1, 10))
    first = evaluate_detection(
        (frame(truth),),
        (frame(predictions),),
        iou_threshold=0.3,
    )
    second = evaluate_detection(
        (frame(tuple(reversed(truth))),),
        (frame(predictions),),
        iou_threshold=0.3,
    )
    assert first == second
