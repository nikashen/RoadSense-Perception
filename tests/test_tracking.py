from __future__ import annotations

import pytest

from roadsense.contracts import BoxXYXY, Detection, FrameRecord, ImageSize
from roadsense.metrics import evaluate_tracking
from roadsense.tracking import IoUTracker

SIZE = ImageSize(width=100, height=100)


def _detection(x: float, *, track_id: int | None = None, score: float = 1.0) -> Detection:
    return Detection(
        category_id=1,
        score=score,
        track_id=track_id,
        bbox=BoxXYXY(x_min=x, y_min=10, x_max=x + 20, y_max=30),
    )


def _frame(index: int, detections: tuple[Detection, ...]) -> FrameRecord:
    return FrameRecord(
        frame_index=index,
        timestamp_ms=index * 100,
        image_size=SIZE,
        detections=detections,
    )


def test_tracker_preserves_identity_across_motion() -> None:
    tracker = IoUTracker(iou_threshold=0.3)
    first = tracker.update((_detection(10),))
    second = tracker.update((_detection(12),))
    assert first[0].track_id == second[0].track_id == 1


def test_tracker_starts_new_identity_after_expiry() -> None:
    tracker = IoUTracker(iou_threshold=0.3, max_age=0)
    assert tracker.update((_detection(10),))[0].track_id == 1
    assert tracker.update(()) == ()
    assert tracker.update((_detection(10),))[0].track_id == 2


def test_tracker_filters_low_scores() -> None:
    tracker = IoUTracker(min_score=0.5)
    assert tracker.update((_detection(10, score=0.2),)) == ()


def test_tracker_reset_restarts_ids() -> None:
    tracker = IoUTracker()
    tracker.update((_detection(10),))
    tracker.reset()
    assert tracker.update((_detection(50),))[0].track_id == 1


def test_perfect_tracking_metrics() -> None:
    frames = (
        _frame(0, (_detection(10, track_id=1),)),
        _frame(1, (_detection(12, track_id=1),)),
    )
    report = evaluate_tracking(frames, frames)
    assert report["mota"] == 1.0
    assert report["identity_f1"] == 1.0
    assert report["identity_switches"] == 0


def test_tracking_counts_identity_switch() -> None:
    truth = (
        _frame(0, (_detection(10, track_id=1),)),
        _frame(1, (_detection(12, track_id=1),)),
    )
    prediction = (
        _frame(0, (_detection(10, track_id=8),)),
        _frame(1, (_detection(12, track_id=9),)),
    )
    report = evaluate_tracking(truth, prediction)
    assert report["identity_switches"] == 1
    assert report["mota"] == 0.5


def test_tracking_requires_ids() -> None:
    with pytest.raises(ValueError, match="track_id"):
        evaluate_tracking((_frame(0, (_detection(10),)),), (_frame(0, (_detection(10),)),))


def test_tracker_rejects_bad_configuration() -> None:
    with pytest.raises(ValueError):
        IoUTracker(iou_threshold=-0.1)
    with pytest.raises(ValueError):
        IoUTracker(max_age=-1)
