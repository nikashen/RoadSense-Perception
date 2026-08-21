"""Strict public contracts shared by metrics, adapters, and the demo service."""

from __future__ import annotations

import math
import re
from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SHA256 = re.compile(r"^[0-9a-f]{64}$")


class StringEnum(str, Enum):
    """String-valued enum that serializes without custom hooks."""


class TaskKind(StringEnum):
    DETECTION = "detection"
    SEGMENTATION = "segmentation"
    TRACKING = "tracking"


class EvidenceLevel(StringEnum):
    FIXTURE = "fixture"
    DEVELOPMENT = "development"
    FROZEN_EVALUATION = "frozen_evaluation"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ImageSize(StrictModel):
    width: Annotated[int, Field(ge=1, le=100_000)]
    height: Annotated[int, Field(ge=1, le=100_000)]


class BoxXYXY(StrictModel):
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    @model_validator(mode="after")
    def validate_geometry(self) -> BoxXYXY:
        values = (self.x_min, self.y_min, self.x_max, self.y_max)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("box coordinates must be finite")
        if self.x_max <= self.x_min or self.y_max <= self.y_min:
            raise ValueError("box must have positive width and height")
        return self

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min

    @property
    def area(self) -> float:
        return self.width * self.height


class Detection(StrictModel):
    category_id: Annotated[int, Field(ge=0, le=1_000_000)]
    score: Annotated[float, Field(ge=0.0, le=1.0)] = 1.0
    bbox: BoxXYXY
    track_id: Annotated[int, Field(ge=0)] | None = None
    label: Annotated[str, Field(min_length=1, max_length=128)] | None = None

    @field_validator("score")
    @classmethod
    def score_must_be_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("score must be finite")
        return value


class FrameRecord(StrictModel):
    frame_index: Annotated[int, Field(ge=0)]
    timestamp_ms: Annotated[int, Field(ge=0)]
    image_size: ImageSize
    detections: tuple[Detection, ...] = ()

    @model_validator(mode="after")
    def detections_must_fit_image(self) -> FrameRecord:
        for detection in self.detections:
            box = detection.bbox
            if (
                box.x_min < 0
                or box.y_min < 0
                or box.x_max > self.image_size.width
                or box.y_max > self.image_size.height
            ):
                raise ValueError("detection box lies outside the image")
        return self


class DatasetManifest(StrictModel):
    schema_version: Literal["roadsense.dataset-manifest/v1"]
    dataset_name: Annotated[str, Field(min_length=1, max_length=256)]
    source_url: Annotated[str, Field(min_length=1, max_length=2_048)]
    license_id: Annotated[str, Field(min_length=1, max_length=256)]
    tasks: tuple[TaskKind, ...]
    splits: dict[str, Annotated[str, Field(min_length=1, max_length=512)]]
    content_sha256: Annotated[str, Field(pattern=SHA256.pattern)]
    evaluation_authorized: bool
    frozen: bool
    notes: Annotated[str, Field(max_length=2_000)] = ""

    @model_validator(mode="after")
    def validate_manifest(self) -> DatasetManifest:
        if not self.tasks or len(set(self.tasks)) != len(self.tasks):
            raise ValueError("tasks must be non-empty and unique")
        if not self.splits or any(not key.strip() for key in self.splits):
            raise ValueError("splits must be non-empty with named entries")
        if self.frozen and not self.evaluation_authorized:
            raise ValueError("a frozen evaluation must be explicitly authorized")
        return self


class EvaluationReport(StrictModel):
    schema_version: Literal["roadsense.evaluation-report/v1"]
    protocol_id: Annotated[str, Field(min_length=1, max_length=256)]
    evidence_level: EvidenceLevel
    dataset_manifest_sha256: Annotated[str, Field(pattern=SHA256.pattern)] | None = None
    evaluation_authorized: bool
    frozen: bool
    metrics: dict[str, float]
    claim_boundary: Annotated[str, Field(min_length=1, max_length=2_000)]

    @model_validator(mode="after")
    def validate_report(self) -> EvaluationReport:
        if any(not math.isfinite(value) for value in self.metrics.values()):
            raise ValueError("metrics must contain finite values")
        if self.evidence_level is EvidenceLevel.FROZEN_EVALUATION:
            if not self.evaluation_authorized or not self.frozen:
                raise ValueError("frozen evidence requires authorization and a frozen report")
            if self.dataset_manifest_sha256 is None:
                raise ValueError("frozen evidence requires a dataset manifest hash")
        if self.frozen and not self.evaluation_authorized:
            raise ValueError("frozen reports must be authorized")
        return self
