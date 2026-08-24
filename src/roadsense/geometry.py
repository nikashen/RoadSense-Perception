"""Geometry and deterministic association primitives."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real

import numpy as np
from numpy.typing import NDArray

from roadsense.contracts import BoxXYXY, Detection


@dataclass(frozen=True, slots=True)
class Match:
    left_index: int
    right_index: int
    iou: float


@dataclass(frozen=True, slots=True)
class MatchResult:
    matches: tuple[Match, ...]
    unmatched_left: tuple[int, ...]
    unmatched_right: tuple[int, ...]


def canonical_detection_key(
    detection: Detection,
) -> tuple[int, float, float, float, float, float, int, str]:
    """Return the protocol tie-break key for one detection.

    Geometry and semantic identity are preferred over container position so
    equivalent detector output permutations produce the same association.
    Exact duplicates remain interchangeable by definition.
    """

    box = detection.bbox
    return (
        detection.category_id,
        box.x_min,
        box.y_min,
        box.x_max,
        box.y_max,
        detection.score,
        detection.track_id if detection.track_id is not None else -1,
        detection.label or "",
    )


def validate_iou_threshold(value: object) -> float:
    """Return a finite IoU threshold or fail with a typed boundary error.

    ``bool`` is a subclass of ``int`` in Python, so a plain chained comparison
    would accidentally accept ``True`` as a threshold of one.  Reject it and
    non-numeric values explicitly at the boundary.
    """

    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("iou_threshold must be a real number in [0, 1]")
    threshold = float(value)
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("iou_threshold must be a finite real number in [0, 1]")
    return threshold


def intersection_over_union(left: BoxXYXY, right: BoxXYXY) -> float:
    # Normalize coordinates before subtraction/multiplication.  Box contracts
    # guarantee finite coordinates, but values near the float limit can still
    # overflow in ``x_max - x_min`` or ``width * height``.  IoU is invariant
    # under a common positive scale, so this preserves the mathematically
    # correct ratio instead of returning zero or NaN on such inputs.
    scale = max(
        abs(left.x_min),
        abs(left.y_min),
        abs(left.x_max),
        abs(left.y_max),
        abs(right.x_min),
        abs(right.y_min),
        abs(right.x_max),
        abs(right.y_max),
    )
    if scale == 0.0:
        scale = 1.0
    left_x_min, left_y_min = left.x_min / scale, left.y_min / scale
    left_x_max, left_y_max = left.x_max / scale, left.y_max / scale
    right_x_min, right_y_min = right.x_min / scale, right.y_min / scale
    right_x_max, right_y_max = right.x_max / scale, right.y_max / scale
    x_min = max(left_x_min, right_x_min)
    y_min = max(left_y_min, right_y_min)
    x_max = min(left_x_max, right_x_max)
    y_max = min(left_y_max, right_y_max)
    intersection = max(0.0, x_max - x_min) * max(0.0, y_max - y_min)
    left_area = (left_x_max - left_x_min) * (left_y_max - left_y_min)
    right_area = (right_x_max - right_x_min) * (right_y_max - right_y_min)
    union = left_area + right_area - intersection
    if union <= 0.0:
        return 0.0
    # Floating-point cancellation can produce a ratio a few ulps outside the
    # mathematical [0, 1] interval; keep downstream thresholds and reports
    # within their declared contract.
    return min(1.0, max(0.0, intersection / union))


def pairwise_iou(left: tuple[Detection, ...], right: tuple[Detection, ...]) -> NDArray[np.float64]:
    matrix = np.zeros((len(left), len(right)), dtype=np.float64)
    for left_index, left_detection in enumerate(left):
        for right_index, right_detection in enumerate(right):
            matrix[left_index, right_index] = intersection_over_union(
                left_detection.bbox, right_detection.bbox
            )
    return matrix


def greedy_iou_match(
    left: tuple[Detection, ...],
    right: tuple[Detection, ...],
    *,
    iou_threshold: float = 0.5,
    class_aware: bool = True,
) -> MatchResult:
    iou_threshold = validate_iou_threshold(iou_threshold)
    candidates: list[
        tuple[
            float,
            tuple[int, float, float, float, float, float, int, str],
            tuple[int, float, float, float, float, float, int, str],
            int,
            int,
        ]
    ] = []
    for left_index, left_detection in enumerate(left):
        for right_index, right_detection in enumerate(right):
            if class_aware and left_detection.category_id != right_detection.category_id:
                continue
            iou = intersection_over_union(left_detection.bbox, right_detection.bbox)
            if iou >= iou_threshold:
                candidates.append(
                    (
                        -iou,
                        canonical_detection_key(left_detection),
                        canonical_detection_key(right_detection),
                        left_index,
                        right_index,
                    )
                )
    candidates.sort()
    used_left: set[int] = set()
    used_right: set[int] = set()
    matches: list[Match] = []
    for negative_iou, _left_key, _right_key, left_index, right_index in candidates:
        if left_index in used_left or right_index in used_right:
            continue
        used_left.add(left_index)
        used_right.add(right_index)
        matches.append(Match(left_index, right_index, -negative_iou))
    return MatchResult(
        matches=tuple(matches),
        unmatched_left=tuple(index for index in range(len(left)) if index not in used_left),
        unmatched_right=tuple(index for index in range(len(right)) if index not in used_right),
    )
