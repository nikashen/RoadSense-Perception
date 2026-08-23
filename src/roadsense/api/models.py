"""Strict response contracts for the local fixture API."""

from __future__ import annotations

import math
from typing import Any, Literal, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictFloat,
    StrictInt,
    field_validator,
    model_validator,
)

from roadsense.contracts import (
    SHA256,
    EvidenceLevel,
    _strict_metrics,
    freeze_value,
    validate_bool,
)
from roadsense.evidence import compute_report_id

EXPECTED_DEMO_CADENCE_MS = 100


class APIModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _assert_finite_json(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("metrics must contain finite JSON numbers")
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError("metrics object keys must be non-empty strings")
            _assert_finite_json(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _assert_finite_json(item)


class HealthResponse(APIModel):
    status: Literal["ok"]
    service: Literal["roadsense-perception"]
    version: str = Field(min_length=1)
    runtime: Literal["deterministic_geometric_fixture"]


class ReadinessResponse(APIModel):
    status: Literal["ready"]
    service_mode: Literal["fixture_replay"]
    verification_level: Literal["fixture"]
    model_loaded: Literal[False]
    benchmark_claim_available: Literal[False]


class SizeResponse(APIModel):
    width: StrictInt = Field(ge=1, le=100_000)
    height: StrictInt = Field(ge=1, le=100_000)


class DemoObjectResponse(APIModel):
    id: str = Field(min_length=1)
    track_id: StrictInt = Field(ge=0)
    label: str = Field(min_length=1)
    category_id: StrictInt = Field(ge=0)
    confidence: StrictFloat = Field(ge=0.0, le=1.0)
    bbox: tuple[StrictFloat, StrictFloat, StrictFloat, StrictFloat]

    @model_validator(mode="after")
    def validate_box(self) -> DemoObjectResponse:
        x, y, width, height = self.bbox
        if not all(math.isfinite(value) for value in self.bbox):
            raise ValueError("bbox values must be finite")
        if width <= 0 or height <= 0:
            raise ValueError("bbox must have positive dimensions")
        if x < 0 or y < 0 or x + width > 960 or y + height > 540:
            raise ValueError("bbox must fit the 960x540 display canvas")
        return self


class DemoSegmentResponse(APIModel):
    label: str = Field(min_length=1)
    category_id: StrictInt = Field(ge=0)
    polygon: tuple[tuple[StrictFloat, StrictFloat], ...] = Field(min_length=3)
    confidence: StrictFloat = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_polygon(self) -> DemoSegmentResponse:
        if any(
            not math.isfinite(value) or value < 0 or value > limit
            for x, y in self.polygon
            for value, limit in ((x, 960), (y, 540))
        ):
            raise ValueError("polygon coordinates must fit the 960x540 display canvas")
        return self


class DemoFrameResponse(APIModel):
    id: str = Field(min_length=1)
    frame_index: StrictInt = Field(ge=0)
    timestamp_ms: StrictInt = Field(ge=0)
    objects: tuple[DemoObjectResponse, ...]
    segments: tuple[DemoSegmentResponse, ...]


class CategoryResponse(APIModel):
    id: StrictInt = Field(ge=0)
    label: str = Field(min_length=1)
    color: str = Field(pattern=r"^#[0-9a-fA-F]{6}$")


class DemoEvidenceResponse(APIModel):
    level: Literal["fixture"]
    evaluation_authorized: Literal[False]
    frozen: Literal[False]
    benchmark_claim_available: Literal[False]
    claim_boundary: str = Field(min_length=1)


class DemoResponse(APIModel):
    schema_version: Literal["roadsense.demo/v1"]
    source: Literal["deterministic_geometric_fixture"]
    fixture_id: str = Field(min_length=1)
    fps: StrictFloat = Field(gt=0)
    cadence_ms: StrictInt = Field(gt=0)
    image_size: SizeResponse
    canvas: SizeResponse
    categories: tuple[CategoryResponse, ...]
    segmentation_categories: tuple[CategoryResponse, ...]
    frames: tuple[DemoFrameResponse, ...] = Field(min_length=2)
    metrics: dict[str, Any]
    evidence: DemoEvidenceResponse

    @field_validator("metrics", mode="after")
    @classmethod
    def freeze_metrics(cls, value: dict[str, Any]) -> dict[str, Any]:
        return cast(dict[str, Any], freeze_value(value))

    @model_validator(mode="after")
    def validate_sequence(self) -> DemoResponse:
        if self.canvas.width != 960 or self.canvas.height != 540:
            raise ValueError("demo canvas must be 960x540")
        if self.image_size.width != 640 or self.image_size.height != 360:
            raise ValueError("demo image_size must be 640x360")
        if self.cadence_ms != EXPECTED_DEMO_CADENCE_MS:
            raise ValueError(f"demo fixture cadence_ms must be {EXPECTED_DEMO_CADENCE_MS}")
        if not math.isclose(self.fps * self.cadence_ms, 1000.0, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError("fps and cadence_ms must describe the same cadence")
        if set(self.metrics) != {"detection", "segmentation", "tracking"}:
            raise ValueError("demo metrics must contain detection, segmentation, and tracking")
        _assert_finite_json(self.metrics)
        if len(self.frames) != 24:
            raise ValueError("demo fixture must contain exactly 24 frames")
        detection_category_ids = {category.id for category in self.categories}
        segmentation_category_ids = {category.id for category in self.segmentation_categories}
        if len(detection_category_ids) != len(self.categories):
            raise ValueError("detection category IDs must be unique")
        if len(segmentation_category_ids) != len(self.segmentation_categories):
            raise ValueError("segmentation category IDs must be unique")
        for index, frame in enumerate(self.frames):
            if frame.frame_index != index:
                raise ValueError("demo frame indices must be contiguous and ordered")
            if frame.timestamp_ms != index * self.cadence_ms:
                raise ValueError("demo timestamps must follow cadence_ms from frame zero")
            if any(item.category_id not in detection_category_ids for item in frame.objects):
                raise ValueError("frame object category is not in the detection ontology")
            if any(item.category_id not in segmentation_category_ids for item in frame.segments):
                raise ValueError("frame segment category is not in the segmentation ontology")
            track_ids = [item.track_id for item in frame.objects]
            if len(track_ids) != len(set(track_ids)):
                raise ValueError("frame object track IDs must be unique")
        return self


class ReportResponse(APIModel):
    schema_version: Literal["roadsense.evaluation-report/v1"]
    protocol_id: str = Field(min_length=1)
    evidence_level: EvidenceLevel
    dataset_manifest_sha256: str | None = Field(default=None, pattern=SHA256.pattern)
    evaluation_authorized: bool
    frozen: bool
    metrics: dict[str, StrictFloat]
    claim_boundary: str = Field(min_length=1)
    report_id: str = Field(pattern=r"^[0-9a-f]{16}$")
    details: dict[str, Any]

    @field_validator("evaluation_authorized", "frozen", mode="before")
    @classmethod
    def flags_must_be_boolean(cls, value: object) -> bool:
        return validate_bool(value)

    @field_validator("metrics", mode="before")
    @classmethod
    def metrics_must_be_strict_numbers(cls, value: object) -> dict[str, float]:
        return _strict_metrics(value)

    @field_validator("metrics", "details", mode="after")
    @classmethod
    def freeze_report_containers(cls, value: dict[str, Any]) -> dict[str, Any]:
        return cast(dict[str, Any], freeze_value(value))

    @model_validator(mode="after")
    def validate_report_identity(self) -> ReportResponse:
        if not self.metrics or any(not key.strip() for key in self.metrics):
            raise ValueError("report metrics must be named and non-empty")
        if any(not math.isfinite(value) for value in self.metrics.values()):
            raise ValueError("report metrics must be finite")
        _assert_finite_json(self.details)
        if self.evidence_level is EvidenceLevel.FIXTURE and (
            self.evaluation_authorized or self.frozen
        ):
            raise ValueError("fixture reports cannot be authorized or frozen")
        if self.evidence_level is EvidenceLevel.FROZEN_EVALUATION:
            if not self.evaluation_authorized or not self.frozen:
                raise ValueError("frozen evidence requires authorization and a frozen report")
            if self.dataset_manifest_sha256 is None:
                raise ValueError("frozen evidence requires a dataset manifest hash")
        if self.frozen and not self.evaluation_authorized:
            raise ValueError("frozen reports must be authorized")
        if self.frozen and self.evidence_level is not EvidenceLevel.FROZEN_EVALUATION:
            raise ValueError("frozen reports must use frozen_evaluation evidence")
        payload = self.model_dump(mode="json")
        if self.report_id != compute_report_id(payload):
            raise ValueError("report_id does not bind the complete report payload")
        return self
