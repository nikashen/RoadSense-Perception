"""Strict response contracts for the local fixture API."""

from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from roadsense.contracts import EvidenceLevel


class APIModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class HealthResponse(APIModel):
    status: Literal["ok"]
    service: Literal["roadsense-perception"]
    version: str = Field(min_length=1)
    runtime: Literal["deterministic_geometric_fixture"]


class ReadinessResponse(APIModel):
    status: Literal["ready"]
    verification_level: Literal["fixture"]
    model_loaded: Literal[False]
    benchmark_claim_available: Literal[False]


class SizeResponse(APIModel):
    width: int = Field(ge=1, le=100_000)
    height: int = Field(ge=1, le=100_000)


class DemoObjectResponse(APIModel):
    id: str = Field(min_length=1)
    track_id: int = Field(ge=0)
    label: str = Field(min_length=1)
    category_id: int = Field(ge=0)
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: tuple[float, float, float, float]

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
    category_id: int = Field(ge=0)
    polygon: tuple[tuple[float, float], ...] = Field(min_length=3)
    confidence: float = Field(ge=0.0, le=1.0)

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
    frame_index: int = Field(ge=0)
    timestamp_ms: int = Field(ge=0)
    objects: tuple[DemoObjectResponse, ...]
    segments: tuple[DemoSegmentResponse, ...]


class CategoryResponse(APIModel):
    id: int = Field(ge=0)
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
    fps: float = Field(gt=0)
    cadence_ms: int = Field(gt=0)
    image_size: SizeResponse
    canvas: SizeResponse
    categories: tuple[CategoryResponse, ...]
    frames: tuple[DemoFrameResponse, ...] = Field(min_length=2)
    metrics: dict[str, Any]
    evidence: DemoEvidenceResponse

    @model_validator(mode="after")
    def validate_sequence(self) -> DemoResponse:
        if self.canvas.width != 960 or self.canvas.height != 540:
            raise ValueError("demo canvas must be 960x540")
        for index, frame in enumerate(self.frames):
            if frame.frame_index != index:
                raise ValueError("demo frame indices must be contiguous and ordered")
            if index and frame.timestamp_ms < self.frames[index - 1].timestamp_ms:
                raise ValueError("demo timestamps must be monotonic")
        return self


class ReportResponse(APIModel):
    schema_version: Literal["roadsense.evaluation-report/v1"]
    protocol_id: str = Field(min_length=1)
    evidence_level: EvidenceLevel
    dataset_manifest_sha256: str | None = None
    evaluation_authorized: bool
    frozen: bool
    metrics: dict[str, float]
    claim_boundary: str = Field(min_length=1)
    report_id: str = Field(min_length=1)
    details: dict[str, Any]
