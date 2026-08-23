"""Dependency-light deterministic IoU tracker used as an explicit baseline."""

from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import SupportsIndex, cast

from roadsense.contracts import BoxXYXY, Detection
from roadsense.geometry import canonical_detection_key, greedy_iou_match, validate_iou_threshold


@dataclass(slots=True)
class _Track:
    track_id: int
    detection: Detection
    age: int = 0
    hits: int = 1


class IoUTracker:
    def __init__(
        self,
        *,
        iou_threshold: float = 0.3,
        max_age: int = 2,
        min_score: float = 0.25,
    ) -> None:
        self.iou_threshold = validate_iou_threshold(iou_threshold)
        if isinstance(max_age, bool):
            raise TypeError("max_age must be a non-negative integer")
        try:
            normalized_max_age = operator.index(cast(SupportsIndex, max_age))
        except TypeError as exc:
            raise TypeError("max_age must be a non-negative integer") from exc
        if normalized_max_age < 0:
            raise ValueError("max_age must be a non-negative integer")
        self.max_age = normalized_max_age
        self.min_score = validate_iou_threshold(min_score)
        self._tracks: list[_Track] = []
        self._next_track_id = 1

    def reset(self) -> None:
        self._tracks.clear()
        self._next_track_id = 1

    def update(self, detections: tuple[Detection, ...]) -> tuple[Detection, ...]:
        filtered = tuple(detection for detection in detections if detection.score >= self.min_score)
        previous = tuple(track.detection for track in self._tracks)
        result = greedy_iou_match(
            previous,
            filtered,
            iou_threshold=self.iou_threshold,
            class_aware=True,
        )
        output_by_detection: dict[int, Detection] = {}
        matched_tracks: set[int] = set()
        for match in result.matches:
            track = self._tracks[match.left_index]
            detection = filtered[match.right_index]
            tracked = Detection(
                category_id=detection.category_id,
                score=detection.score,
                bbox=BoxXYXY(**detection.bbox.model_dump()),
                track_id=track.track_id,
                label=detection.label,
            )
            track.detection = tracked
            track.age = 0
            track.hits += 1
            matched_tracks.add(match.left_index)
            output_by_detection[match.right_index] = tracked
        survivors: list[_Track] = []
        for index, track in enumerate(self._tracks):
            if index not in matched_tracks:
                track.age += 1
            if track.age <= self.max_age:
                survivors.append(track)
        self._tracks = survivors
        for detection_index in sorted(
            result.unmatched_right,
            key=lambda index: canonical_detection_key(filtered[index]),
        ):
            detection = filtered[detection_index]
            tracked = Detection(
                category_id=detection.category_id,
                score=detection.score,
                bbox=BoxXYXY(**detection.bbox.model_dump()),
                track_id=self._next_track_id,
                label=detection.label,
            )
            self._tracks.append(_Track(self._next_track_id, tracked))
            self._next_track_id += 1
            output_by_detection[detection_index] = tracked
        return tuple(output_by_detection[index] for index in sorted(output_by_detection))
