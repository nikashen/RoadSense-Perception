"""Strict public contracts shared by metrics, adapters, and the demo service."""

from __future__ import annotations

import math
import numbers
import re
from collections.abc import Mapping
from enum import Enum
from typing import Annotated, Any, Literal, NoReturn, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SHA256 = re.compile(r"^[0-9a-f]{64}$")


class FrozenDict(dict[str, Any]):
    """A JSON-compatible mapping that rejects in-place mutation."""

    def _blocked(self, *args: object, **kwargs: object) -> NoReturn:
        del args, kwargs
        raise TypeError("mapping is immutable")

    __setitem__ = _blocked
    __delitem__ = _blocked
    clear = _blocked
    pop = _blocked
    popitem = _blocked
    setdefault = _blocked
    update = _blocked
    __ior__ = _blocked


def freeze_value(value: Any) -> Any:
    """Recursively freeze JSON containers while preserving JSON serialization."""

    if isinstance(value, dict):
        return FrozenDict({key: freeze_value(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(freeze_value(item) for item in value)
    if isinstance(value, set):
        return frozenset(freeze_value(item) for item in value)
    return value


def strict_bool(value: object) -> bool:
    """Reject numeric/string coercion for evidence authorization flags."""

    if not isinstance(value, bool):
        raise TypeError("value must be a boolean")
    return value


def validate_bool(value: object) -> bool:
    """Pydantic-friendly wrapper for :func:`strict_bool`."""

    try:
        return strict_bool(value)
    except TypeError as exc:
        raise ValueError(str(exc)) from exc


def _strict_integer(value: object, field_name: str) -> int:
    """Accept Python/NumPy integral values without coercing strings or bools.

    Pydantic's default ``int`` validator intentionally performs convenience
    coercions (for example ``"3"`` and ``True`` become integers).  That is a
    dangerous property for frame indices, image dimensions, and class/track
    IDs in an evaluation bundle: a malformed upstream record would silently
    change meaning before metrics run.  ``numbers.Integral`` keeps normal
    Python integers and NumPy integer scalars compatible while rejecting
    floating-point, string, and boolean values.
    """

    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{field_name} must be an integer")  # noqa: TRY004 - Pydantic wraps it.
    return int(value)


def _strict_real(value: object, field_name: str) -> float:
    """Accept Python/NumPy real values without coercing strings or bools."""

    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{field_name} must be a number")  # noqa: TRY004 - Pydantic wraps it.
    return float(value)


def _strict_metrics(value: object, field_name: str = "metrics") -> dict[str, float]:
    """Validate report metrics without Pydantic's lossy numeric coercions.

    Reports are evidence artifacts.  Accepting ``"0.9"`` or ``True`` here
    would make malformed JSON look like a measured score and would change the
    canonical report hash after validation.  JSON numbers arrive as Python
    ``int``/``float`` values, while NumPy real scalars remain useful to local
    callers, so reuse the strict real boundary and normalize only after it has
    been checked.
    """

    if not isinstance(value, Mapping):
        raise ValueError(  # noqa: TRY004 - validators expose malformed JSON as validation errors.
            f"{field_name} must be an object of finite numbers"
        )
    result: dict[str, float] = {}
    for key, item in value.items():
        if (
            not isinstance(key, str)
            or not key.strip()
            or key != key.strip()
            or any(ord(character) < 32 or ord(character) == 127 for character in key)
        ):
            raise ValueError(f"{field_name} keys must be non-empty strings")
        result[key] = _strict_real(item, f"{field_name}.{key}")
        if not math.isfinite(result[key]):
            raise ValueError(f"{field_name}.{key} must be finite")
    return result


def _strict_text(value: object, field_name: str) -> str:
    """Reject blank, padded, or control-character protocol text."""

    if not isinstance(value, str):
        raise ValueError(  # noqa: TRY004 - validators expose malformed JSON as validation errors.
            f"{field_name} must be a string"
        )
    if not value.strip() or value != value.strip():
        raise ValueError(f"{field_name} must not be blank or padded")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ValueError(f"{field_name} must not contain control characters")
    return value


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

    @field_validator("width", "height", mode="before")
    @classmethod
    def dimensions_must_be_integers(cls, value: object, info: Any) -> int:
        return _strict_integer(value, str(info.field_name))


class BoxXYXY(StrictModel):
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    @field_validator("x_min", "y_min", "x_max", "y_max", mode="before")
    @classmethod
    def coordinates_must_be_numeric(cls, value: object, info: Any) -> float:
        return _strict_real(value, str(info.field_name))

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

    @field_validator("category_id", mode="before")
    @classmethod
    def category_id_must_be_integer(cls, value: object) -> int:
        return _strict_integer(value, "category_id")

    @field_validator("track_id", mode="before")
    @classmethod
    def track_id_must_be_integer(cls, value: object) -> int | None:
        return None if value is None else _strict_integer(value, "track_id")

    @field_validator("score", mode="before")
    @classmethod
    def score_must_be_numeric(cls, value: object) -> float:
        return _strict_real(value, "score")

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

    @field_validator("frame_index", "timestamp_ms", mode="before")
    @classmethod
    def temporal_fields_must_be_integers(cls, value: object, info: Any) -> int:
        return _strict_integer(value, str(info.field_name))

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

    @field_validator("dataset_name", "source_url", "license_id", mode="before")
    @classmethod
    def identity_text_must_be_clean(cls, value: object, info: Any) -> str:
        return _strict_text(value, str(info.field_name))

    @field_validator("evaluation_authorized", "frozen", mode="before")
    @classmethod
    def flags_must_be_boolean(cls, value: object) -> bool:
        return validate_bool(value)

    @field_validator("splits", mode="before")
    @classmethod
    def split_text_must_be_clean(cls, value: object) -> object:
        if not isinstance(value, Mapping):
            return value
        return {
            _strict_text(key, "split name"): _strict_text(item, f"splits[{key}]")
            for key, item in value.items()
        }

    @field_validator("splits", mode="after")
    @classmethod
    def freeze_splits(cls, value: dict[str, str]) -> dict[str, str]:
        for key, item in value.items():
            _strict_text(key, "split name")
            _strict_text(item, f"splits[{key}]")
        return cast(dict[str, str], freeze_value(value))

    @model_validator(mode="after")
    def validate_manifest(self) -> DatasetManifest:
        if not self.tasks or len(set(self.tasks)) != len(self.tasks):
            raise ValueError("tasks must be non-empty and unique")
        if not self.splits or any(not key.strip() for key in self.splits):
            raise ValueError("splits must be non-empty with named entries")
        if self.content_sha256 == "0" * 64:
            raise ValueError("content_sha256 cannot be an all-zero placeholder")
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

    @field_validator("protocol_id", "claim_boundary", mode="before")
    @classmethod
    def report_text_must_be_clean(cls, value: object, info: Any) -> str:
        return _strict_text(value, str(info.field_name))

    @field_validator("evaluation_authorized", "frozen", mode="before")
    @classmethod
    def flags_must_be_boolean(cls, value: object) -> bool:
        return validate_bool(value)

    @field_validator("metrics", mode="before")
    @classmethod
    def metrics_must_be_strict_numbers(cls, value: object) -> dict[str, float]:
        return _strict_metrics(value)

    @field_validator("metrics", mode="after")
    @classmethod
    def freeze_metrics(cls, value: dict[str, float]) -> dict[str, float]:
        return cast(dict[str, float], freeze_value(value))

    @model_validator(mode="after")
    def validate_report(self) -> EvaluationReport:
        if not self.metrics or any(not key.strip() for key in self.metrics):
            raise ValueError("evaluation reports require named metrics")
        if any(not math.isfinite(value) for value in self.metrics.values()):
            raise ValueError("metrics must contain finite values")
        if self.evidence_level is EvidenceLevel.FIXTURE and (
            self.evaluation_authorized or self.frozen
        ):
            raise ValueError("fixture evidence cannot be authorized or frozen")
        if self.evidence_level is EvidenceLevel.FROZEN_EVALUATION:
            if not self.evaluation_authorized or not self.frozen:
                raise ValueError("frozen evidence requires authorization and a frozen report")
            if self.dataset_manifest_sha256 is None:
                raise ValueError("frozen evidence requires a dataset manifest hash")
        if self.frozen and not self.evaluation_authorized:
            raise ValueError("frozen reports must be authorized")
        if self.frozen and self.evidence_level is not EvidenceLevel.FROZEN_EVALUATION:
            raise ValueError("frozen reports must use frozen_evaluation evidence")
        return self
