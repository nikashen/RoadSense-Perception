"""Deterministic geometric road-scene fixture for CI and Pages."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import cast

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
FIXTURE_ID = "roadsense-city-loop-v4"
VEHICLE_TRACK_IDS = (101, 102)
ROAD_HORIZON_Y = 250.0
ROAD_BOTTOM_Y = 540.0
ROAD_TOP_LEFT_X = 419.0
ROAD_TOP_RIGHT_X = 543.0
ROAD_BOTTOM_LEFT_X = 236.0
ROAD_BOTTOM_RIGHT_X = 779.0
ROAD_TOP_CENTER_X = 481.0
ROAD_BOTTOM_CENTER_X = 503.0


@dataclass(frozen=True, slots=True)
class FixtureBundle:
    truth_frames: tuple[FrameRecord, ...]
    prediction_frames: tuple[FrameRecord, ...]
    truth_masks: NDArray[np.int64]
    prediction_masks: NDArray[np.int64]


def _box(x: float, y: float, width: float, height: float) -> BoxXYXY:
    return BoxXYXY(x_min=x, y_min=y, x_max=x + width, y_max=y + height)


def _lane_vehicle_box(
    frame_index: int,
    *,
    lane: str,
    start_bottom: float,
    end_bottom: float,
    start_width: float,
    end_width: float,
    lane_fraction: float,
) -> BoxXYXY:
    """Project an approaching vehicle along a perspective lane centerline."""

    progress = frame_index / (FRAME_COUNT - 1)
    bottom = start_bottom + (end_bottom - start_bottom) * progress
    width = start_width + (end_width - start_width) * progress
    height = width * 0.58

    # These display-space edges match the road polygon rendered by app.js.
    road_progress = (bottom - ROAD_HORIZON_Y) / (ROAD_BOTTOM_Y - ROAD_HORIZON_Y)
    road_center = ROAD_TOP_CENTER_X + (ROAD_BOTTOM_CENTER_X - ROAD_TOP_CENTER_X) * road_progress
    left_edge = ROAD_TOP_LEFT_X + (ROAD_BOTTOM_LEFT_X - ROAD_TOP_LEFT_X) * road_progress
    right_edge = ROAD_TOP_RIGHT_X + (ROAD_BOTTOM_RIGHT_X - ROAD_TOP_RIGHT_X) * road_progress
    if lane == "left":
        center_x = road_center - (road_center - left_edge) * lane_fraction
    elif lane == "right":
        center_x = road_center + (right_edge - road_center) * lane_fraction
    else:
        raise ValueError("lane must be 'left' or 'right'")

    scale_x = WIDTH / DEMO_WIDTH
    scale_y = HEIGHT / DEMO_HEIGHT
    return _box(
        (center_x - width / 2.0) * scale_x,
        (bottom - height) * scale_y,
        width * scale_x,
        height * scale_y,
    )


def _truth_detections(frame_index: int) -> tuple[Detection, ...]:
    detections: list[Detection] = [
        Detection(
            category_id=1,
            label="car",
            track_id=101,
            bbox=_lane_vehicle_box(
                frame_index,
                lane="right",
                start_bottom=330.0,
                end_bottom=420.0,
                start_width=72.0,
                end_width=112.0,
                lane_fraction=0.28,
            ),
        )
    ]
    if 2 <= frame_index <= 21:
        detections.append(
            Detection(
                category_id=1,
                label="car",
                track_id=102,
                bbox=_lane_vehicle_box(
                    frame_index,
                    lane="left",
                    start_bottom=305.0,
                    end_bottom=360.0,
                    start_width=55.0,
                    end_width=78.0,
                    lane_fraction=0.28,
                ),
            )
        )
    if 4 <= frame_index <= 18:
        detections.append(
            Detection(
                category_id=2,
                label="pedestrian",
                track_id=201,
                bbox=_box(535 - frame_index * 1.2, 178 + frame_index * 1.2, 24, 68),
            )
        )
    if 9 <= frame_index <= 23:
        detections.append(
            Detection(
                category_id=3,
                label="cyclist",
                track_id=301,
                bbox=_box(82 + frame_index * 5.0, 188, 47, 72),
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
                bbox=_lane_vehicle_box(
                    frame_index,
                    lane="right",
                    start_bottom=286.0,
                    end_bottom=330.0,
                    start_width=42.0,
                    end_width=58.0,
                    lane_fraction=0.62,
                ),
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
        # Keep the raster ontology aligned with the display polygons: road=1,
        # car=2, vulnerable road user=3, sidewalk=4, background=0.
        mask[row, :left] = 4
        mask[row, right:] = 4
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
            num_classes=5,
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


def _road_bounds_at_display_y(y: float) -> tuple[float, float]:
    if not ROAD_HORIZON_Y <= y <= ROAD_BOTTOM_Y:
        raise RuntimeError("vehicle ground contact must lie within the rendered road depth")
    progress = (y - ROAD_HORIZON_Y) / (ROAD_BOTTOM_Y - ROAD_HORIZON_Y)
    left = ROAD_TOP_LEFT_X + (ROAD_BOTTOM_LEFT_X - ROAD_TOP_LEFT_X) * progress
    right = ROAD_TOP_RIGHT_X + (ROAD_BOTTOM_RIGHT_X - ROAD_TOP_RIGHT_X) * progress
    return left, right


def _validate_vehicle_road_placement(objects: list[dict[str, object]]) -> None:
    for item in objects:
        if item.get("label") != "car":
            continue
        bbox = item.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise RuntimeError("vehicle display bbox is invalid")
        if not all(isinstance(value, (int, float)) for value in bbox):
            raise RuntimeError("vehicle display bbox must be numeric")
        x, y, width, height = (float(value) for value in bbox)
        left, right = _road_bounds_at_display_y(y + height)
        if x < left or x + width > right:
            raise RuntimeError("vehicle display bbox leaves the rendered road at ground contact")


def _numeric_bbox(item: dict[str, object]) -> tuple[float, float, float, float]:
    bbox = item.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise RuntimeError("actor display bbox is invalid")
    if not all(isinstance(value, (int, float)) for value in bbox):
        raise RuntimeError("actor display bbox must be numeric")
    numeric_bbox = cast(list[int | float], bbox)
    x, y, width, height = numeric_bbox
    return float(x), float(y), float(width), float(height)


def _validate_actor_separation(objects: list[dict[str, object]]) -> None:
    visible: list[dict[str, object]] = []
    for item in objects:
        confidence = item.get("confidence")
        if isinstance(confidence, (int, float)) and float(confidence) >= 0.5:
            visible.append(item)
    for first, second in combinations(visible, 2):
        labels = {first.get("label"), second.get("label")}
        if "car" not in labels or labels.isdisjoint({"pedestrian", "cyclist"}):
            continue
        first_x, first_y, first_width, first_height = _numeric_bbox(first)
        second_x, second_y, second_width, second_height = _numeric_bbox(second)
        overlap_width = max(
            0.0,
            min(first_x + first_width, second_x + second_width) - max(first_x, second_x),
        )
        overlap_height = max(
            0.0,
            min(first_y + first_height, second_y + second_height) - max(first_y, second_y),
        )
        if overlap_width * overlap_height > 0.0:
            raise RuntimeError("visible vehicle and vulnerable-road-user actors overlap")


def _road_segments() -> list[dict[str, object]]:
    return [
        {
            "label": "road",
            "category_id": 1,
            "polygon": [
                [ROAD_BOTTOM_LEFT_X, ROAD_BOTTOM_Y],
                [ROAD_TOP_LEFT_X, ROAD_HORIZON_Y],
                [ROAD_TOP_RIGHT_X, ROAD_HORIZON_Y],
                [ROAD_BOTTOM_RIGHT_X, ROAD_BOTTOM_Y],
            ],
            "confidence": 0.96,
        },
        {
            "label": "sidewalk",
            "category_id": 4,
            "polygon": [[0, 540], [0, 377], [345, 250], [419, 250], [236, 540]],
            "confidence": 0.88,
        },
        {
            "label": "sidewalk",
            "category_id": 4,
            "polygon": [[543, 250], [615, 250], [960, 349], [960, 540], [779, 540]],
            "confidence": 0.88,
        },
    ]


def build_demo_payload(bundle: FixtureBundle | None = None) -> dict[str, object]:
    selected = bundle or build_fixture_bundle()
    metrics = build_fixture_metrics(selected)
    frames = []
    for frame in selected.prediction_frames:
        objects = [_object_payload(detection) for detection in frame.detections]
        _validate_vehicle_road_placement(objects)
        _validate_actor_separation(objects)
        frames.append(
            {
                "id": f"frame-{frame.frame_index:03d}",
                "frame_index": frame.frame_index,
                "timestamp_ms": frame.timestamp_ms,
                "objects": objects,
                "segments": _road_segments(),
            }
        )
    return {
        "schema_version": "roadsense.demo/v1",
        "source": "deterministic_geometric_fixture",
        "fixture_id": FIXTURE_ID,
        "fps": FPS,
        "cadence_ms": 1000 // FPS,
        "image_size": {"width": WIDTH, "height": HEIGHT},
        "canvas": {"width": DEMO_WIDTH, "height": DEMO_HEIGHT},
        "categories": [
            {"id": 1, "label": "car", "color": "#64d8cb"},
            {"id": 2, "label": "pedestrian", "color": "#ffbc69"},
            {"id": 3, "label": "cyclist", "color": "#c8a7ff"},
        ],
        "segmentation_categories": [
            {"id": 0, "label": "background", "color": "#071411"},
            {"id": 1, "label": "road", "color": "#42e1c3"},
            {"id": 2, "label": "car", "color": "#64d8cb"},
            {"id": 3, "label": "vulnerable road user", "color": "#c8a7ff"},
            {"id": 4, "label": "sidewalk", "color": "#a99aff"},
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


__all__ = [
    "FIXTURE_ID",
    "FRAME_COUNT",
    "VEHICLE_TRACK_IDS",
    "FixtureBundle",
    "build_demo_payload",
    "build_fixture_bundle",
    "build_fixture_metrics",
]
