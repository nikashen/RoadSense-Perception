from __future__ import annotations

import numpy as np
import pytest

from roadsense.contracts import BoxXYXY, Detection, FrameRecord, ImageSize
from roadsense.metrics import evaluate_tracking
from roadsense.tracking import IoUTracker

SIZE = ImageSize(width=100, height=100)


def _detection(
    x: float,
    *,
    track_id: int | None = None,
    score: float = 1.0,
    category_id: int = 1,
) -> Detection:
    return Detection(
        category_id=category_id,
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


def test_tracker_assigns_new_ids_independent_of_detection_order() -> None:
    first_tracker = IoUTracker()
    second_tracker = IoUTracker()
    first = first_tracker.update((_detection(50), _detection(10)))
    second = second_tracker.update((_detection(10), _detection(50)))
    first_by_x = {detection.bbox.x_min: detection.track_id for detection in first}
    second_by_x = {detection.bbox.x_min: detection.track_id for detection in second}
    assert first_by_x == second_by_x == {10: 1, 50: 2}


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
    with pytest.raises((TypeError, ValueError)):
        IoUTracker(max_age=1.5)
    with pytest.raises((TypeError, ValueError)):
        IoUTracker(min_score=True)
    assert IoUTracker(max_age=np.int64(1)).max_age == 1


def test_tracking_rejects_duplicate_track_ids_in_a_frame() -> None:
    duplicate = _frame(0, (_detection(10, track_id=1), _detection(50, track_id=1)))
    with pytest.raises(ValueError, match="unique"):
        evaluate_tracking((duplicate,), (duplicate,))


def test_tracking_rejects_invalid_threshold_even_for_empty_frames() -> None:
    frame = _frame(0, ())
    with pytest.raises(ValueError, match="iou_threshold"):
        evaluate_tracking((frame,), (frame,), iou_threshold=1.1)


def test_tracking_rejects_empty_ground_truth_instead_of_claiming_mota_one() -> None:
    frame = _frame(0, ())
    with pytest.raises(ValueError, match="at least one detection"):
        evaluate_tracking((frame,), (frame,))


@pytest.mark.parametrize("role", ["ground_truth", "prediction"])
def test_tracking_rejects_track_id_category_changes_across_frames(role: str) -> None:
    first = _frame(0, (_detection(10, track_id=1, category_id=1),))
    second = _frame(1, (_detection(12, track_id=1, category_id=2),))
    truth = (first, second)
    prediction = truth
    if role == "prediction":
        prediction = (
            _frame(0, (_detection(10, track_id=8, category_id=1),)),
            _frame(1, (_detection(12, track_id=8, category_id=2),)),
        )
        truth = (
            _frame(0, (_detection(10, track_id=1, category_id=1),)),
            _frame(1, (_detection(12, track_id=1, category_id=1),)),
        )
    with pytest.raises(ValueError, match="changes category"):
        evaluate_tracking(truth, prediction)
