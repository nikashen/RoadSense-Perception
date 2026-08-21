"""Geometry and deterministic association primitives."""

from __future__ import annotations

from dataclasses import dataclass

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


def intersection_over_union(left: BoxXYXY, right: BoxXYXY) -> float:
    x_min = max(left.x_min, right.x_min)
    y_min = max(left.y_min, right.y_min)
    x_max = min(left.x_max, right.x_max)
    y_max = min(left.y_max, right.y_max)
    intersection = max(0.0, x_max - x_min) * max(0.0, y_max - y_min)
    union = left.area + right.area - intersection
    return 0.0 if union <= 0.0 else intersection / union


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
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be in [0, 1]")
    candidates: list[tuple[float, int, int]] = []
    for left_index, left_detection in enumerate(left):
        for right_index, right_detection in enumerate(right):
            if class_aware and left_detection.category_id != right_detection.category_id:
                continue
            iou = intersection_over_union(left_detection.bbox, right_detection.bbox)
            if iou >= iou_threshold:
                candidates.append((-iou, left_index, right_index))
    candidates.sort()
    used_left: set[int] = set()
    used_right: set[int] = set()
    matches: list[Match] = []
    for negative_iou, left_index, right_index in candidates:
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
