"""Deterministic geometric road-scene fixture for CI and Pages."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from roadsense.contracts import BoxXYXY, Detection, FrameRecord, ImageSize
from roadsense.metrics import evaluate_detection, evaluate_segmentation, evaluate_tracking
from roadsense.tracking import IoUTracker

WIDTH = 640
HEIGHT = 360
DEMO_WIDTH = 960
DEMO_HEIGHT = 540
FPS = 10
FRAME_COUNT = 24
SEGMENTATION_WIDTH = 160
SEGMENTATION_HEIGHT = 90

CATEGORY_LABELS = {1: "car", 2: "pedestrian", 3: "cyclist"}


@dataclass(frozen=True, slots=True)
class FixtureBundle:
    truth_frames: tuple[FrameRecord, ...]
    prediction_frames: tuple[FrameRecord, ...]
    truth_masks: NDArray[np.int64]
    prediction_masks: NDArray[np.int64]


def _box(x: float, y: float, width: float, height: float) -> BoxXYXY:
    return BoxXYXY(x_min=x, y_min=y, x_max=x + width, y_max=y + height)


def _truth_detections(frame_index: int) -> tuple[Detection, ...]:
    detections: list[Detection] = [
        Detection(
            category_id=1,
            label="car",
            track_id=101,
            bbox=_box(68 + frame_index * 11.0, 228, 104, 58),
        )
    ]
    if 2 <= frame_index <= 21:
        detections.append(
            Detection(
                category_id=1,
                label="car",
                track_id=102,
                bbox=_box(506 - frame_index * 7.0, 202, 82, 48),
            )
        )
    if 4 <= frame_index <= 18:
        detections.append(
            Detection(
                category_id=2,
                label="pedestrian",
                track_id=201,
                bbox=_box(292 + frame_index * 2.0, 178 + frame_index * 1.2, 24, 68),
            )
        )
    if 9 <= frame_index <= 23:
        detections.append(
            Detection(
                category_id=3,
                label="cyclist",
                track_id=301,
                bbox=_box(92 + frame_index * 7.0, 188, 47, 72),
            )
        )
    return tuple(detections)


def _prediction_detections(frame_index: int, truth: tuple[Detection, ...]) -> tuple[Detection, ...]:
    predictions: list[Detection] = []
    for detection in truth:
        if detection.track_id == 201 and frame_index in {10, 11, 12}:
            continue
        if detection.track_id == 102 and frame_index == 15:
            continue
        jitter_x = ((frame_index + int(detection.track_id or 0)) % 5 - 2) * 1.3
        jitter_y = ((frame_index * 3 + detection.category_id) % 5 - 2) * 0.7
        box = detection.bbox
        predictions.append(
            Detection(
                category_id=detection.category_id,
                label=detection.label,
                score=max(0.52, 0.94 - 0.012 * frame_index - 0.02 * detection.category_id),
                bbox=BoxXYXY(
                    x_min=max(0.0, box.x_min + jitter_x),
                    y_min=max(0.0, box.y_min + jitter_y),
                    x_max=min(float(WIDTH), box.x_max + jitter_x + 1.0),
                    y_max=min(float(HEIGHT), box.y_max + jitter_y - 0.5),
                ),
            )
        )
    if frame_index in {7, 16}:
        predictions.append(
            Detection(
                category_id=1,
                label="car",
                score=0.38,
                bbox=_box(444 - frame_index, 174, 55, 36),
            )
        )
    return tuple(predictions)


def _paint_mask(
    frame_index: int, detections: tuple[Detection, ...], *, predicted: bool
) -> NDArray[np.int64]:
    mask = np.zeros((SEGMENTATION_HEIGHT, SEGMENTATION_WIDTH), dtype=np.int64)
    horizon = 42 + (1 if predicted else 0)
    for row in range(horizon, SEGMENTATION_HEIGHT):
        inset = round((row - horizon) * 0.34)
        left = max(0, 45 - inset)
        right = min(SEGMENTATION_WIDTH, 115 + inset)
        mask[row, left:right] = 1
    for detection in detections:
        if predicted and detection.label == "pedestrian" and frame_index in {10, 11, 12}:
            continue
        scale_x = SEGMENTATION_WIDTH / WIDTH
        scale_y = SEGMENTATION_HEIGHT / HEIGHT
        box = detection.bbox
        left = max(0, int(box.x_min * scale_x))
        top = max(0, int(box.y_min * scale_y))
        right = min(SEGMENTATION_WIDTH, max(left + 1, int(np.ceil(box.x_max * scale_x))))
        bottom = min(SEGMENTATION_HEIGHT, max(top + 1, int(np.ceil(box.y_max * scale_y))))
        category = 2 if detection.category_id == 1 else 3
        mask[top:bottom, left:right] = category
    if predicted and frame_index % 6 == 0:
        mask[48:51, 20:31] = 1
    return mask


def build_fixture_bundle() -> FixtureBundle:
    tracker = IoUTracker(iou_threshold=0.25, max_age=2, min_score=0.25)
    size = ImageSize(width=WIDTH, height=HEIGHT)
    truth_frames: list[FrameRecord] = []
    prediction_frames: list[FrameRecord] = []
    truth_masks: list[NDArray[np.int64]] = []
    prediction_masks: list[NDArray[np.int64]] = []
    for frame_index in range(FRAME_COUNT):
        truth = _truth_detections(frame_index)
        predictions = _prediction_detections(frame_index, truth)
        tracked_predictions = tracker.update(predictions)
        truth_frames.append(
            FrameRecord(
                frame_index=frame_index,
                timestamp_ms=frame_index * (1000 // FPS),
                image_size=size,
                detections=truth,
            )
        )
        prediction_frames.append(
            FrameRecord(
                frame_index=frame_index,
                timestamp_ms=frame_index * (1000 // FPS),
                image_size=size,
                detections=tracked_predictions,
            )
        )
        truth_masks.append(_paint_mask(frame_index, truth, predicted=False))
        prediction_masks.append(_paint_mask(frame_index, predictions, predicted=True))
    return FixtureBundle(
        truth_frames=tuple(truth_frames),
        prediction_frames=tuple(prediction_frames),
        truth_masks=np.stack(truth_masks),
        prediction_masks=np.stack(prediction_masks),
    )


def build_fixture_metrics(bundle: FixtureBundle | None = None) -> dict[str, object]:
    selected = bundle or build_fixture_bundle()
    return {
        "detection": evaluate_detection(selected.truth_frames, selected.prediction_frames),
        "segmentation": evaluate_segmentation(
            selected.truth_masks,
            selected.prediction_masks,
            num_classes=4,
        ),
        "tracking": evaluate_tracking(selected.truth_frames, selected.prediction_frames),
    }


def _object_payload(detection: Detection) -> dict[str, object]:
    box = detection.bbox
    scale_x = DEMO_WIDTH / WIDTH
    scale_y = DEMO_HEIGHT / HEIGHT
    return {
        "id": f"track-{detection.track_id}",
        "track_id": detection.track_id,
        "label": detection.label or CATEGORY_LABELS[detection.category_id],
        "category_id": detection.category_id,
        "confidence": detection.score,
        "bbox": [
            box.x_min * scale_x,
            box.y_min * scale_y,
            box.width * scale_x,
            box.height * scale_y,
        ],
    }


def build_demo_payload(bundle: FixtureBundle | None = None) -> dict[str, object]:
    selected = bundle or build_fixture_bundle()
    metrics = build_fixture_metrics(selected)
    frames = []
    for frame in selected.prediction_frames:
        frames.append(
            {
                "id": f"frame-{frame.frame_index:03d}",
                "frame_index": frame.frame_index,
                "timestamp_ms": frame.timestamp_ms,
                "objects": [_object_payload(detection) for detection in frame.detections],
                "segments": [
                    {
                        "label": "road",
                        "category_id": 1,
                        "polygon": [[0, 540], [285, 255], [675, 255], [960, 540]],
                        "confidence": 0.96,
                    },
                    {
                        "label": "sidewalk",
                        "category_id": 4,
                        "polygon": [[0, 540], [0, 345], [285, 255], [225, 540]],
                        "confidence": 0.88,
                    },
                ],
            }
        )
    return {
        "schema_version": "roadsense.demo/v1",
        "source": "deterministic_geometric_fixture",
        "fixture_id": "roadsense-city-loop-v1",
        "fps": FPS,
        "cadence_ms": 1000 // FPS,
        "image_size": {"width": WIDTH, "height": HEIGHT},
        "canvas": {"width": DEMO_WIDTH, "height": DEMO_HEIGHT},
        "categories": [
            {"id": 1, "label": "car", "color": "#64d8cb"},
            {"id": 2, "label": "pedestrian", "color": "#ffbc69"},
            {"id": 3, "label": "cyclist", "color": "#c8a7ff"},
        ],
        "frames": frames,
        "metrics": metrics,
        "evidence": {
            "level": "fixture",
            "evaluation_authorized": False,
            "frozen": False,
            "benchmark_claim_available": False,
            "claim_boundary": (
                "Synthetic geometry validates contracts, replay, and metric plumbing only; "
                "it is not BDD100K, COCO, MOT, KITTI, or nuScenes evidence."
            ),
        },
    }
