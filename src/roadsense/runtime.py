"""Auditable runtime records for fixture dry-runs and local experiments.

The runtime record deliberately separates *measurement* from *claim*.  A
fixture run can report wall-clock timings, device information, dependency
versions, and content hashes, but it is never promoted to a benchmark claim.
The schema is JSON serialisable and its ``record_id`` binds every field except
the identifier itself.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import math
import os
import platform
import sys
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from pydantic import Field, field_validator, model_validator

from roadsense import __version__
from roadsense.contracts import SHA256, EvidenceLevel, StrictModel, freeze_value, validate_bool
from roadsense.fixture import (
    FRAME_COUNT,
    FixtureBundle,
    build_demo_payload,
    build_fixture_bundle,
    build_fixture_metrics,
)
from roadsense.json_io import canonical_sha256, write_json_atomic

RUNTIME_SCHEMA = "roadsense.runtime-audit/v1"
RECORD_ID_PATTERN = r"^[0-9a-f]{16}$"


def _finite(value: float, field_name: str) -> float:
    if not math.isfinite(value):
        raise ValueError(f"{field_name} must be finite")
    return value


def _strict_integer(value: object, field_name: str) -> int:
    """Reject bool/string/float coercion in audit cardinalities."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")  # noqa: TRY004
    return value


def _strict_number(value: object, field_name: str) -> float:
    """Reject bool/string coercion before Pydantic converts numeric fields."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a number")  # noqa: TRY004
    return float(value)


class RuntimeDevice(StrictModel):
    """Non-sensitive host/runtime information needed to reproduce a run."""

    device: str = Field(min_length=1, max_length=128)
    operating_system: str = Field(min_length=1, max_length=128)
    operating_system_version: str = Field(min_length=1, max_length=256)
    machine: str = Field(min_length=1, max_length=128)
    processor: str = Field(min_length=1, max_length=256)
    python_implementation: str = Field(min_length=1, max_length=64)
    python_version: str = Field(min_length=1, max_length=64)
    cpu_count: int | None = Field(default=None, ge=1, le=1_000_000)

    @field_validator("cpu_count", mode="before")
    @classmethod
    def cpu_count_must_be_integer(cls, value: object) -> object:
        return None if value is None else _strict_integer(value, "cpu_count")


class RuntimeInput(StrictModel):
    """Identity and cardinality of the input consumed by a run."""

    source: str = Field(min_length=1, max_length=256)
    fixture_id: str | None = Field(default=None, min_length=1, max_length=256)
    frame_count: int = Field(ge=1, le=10_000_000)
    iterations: int = Field(ge=1, le=10_000)
    payload_sha256: str = Field(pattern=SHA256.pattern)

    @field_validator("frame_count", "iterations", mode="before")
    @classmethod
    def cardinalities_must_be_integer(cls, value: object, info: Any) -> int:
        return _strict_integer(value, info.field_name)

    @model_validator(mode="after")
    def validate_fixture_identity(self) -> RuntimeInput:
        if self.source == "deterministic_geometric_fixture" and self.fixture_id is None:
            raise ValueError("fixture input requires fixture_id")
        if self.payload_sha256 == "0" * 64:
            raise ValueError("input payload hash cannot be an all-zero placeholder")
        return self


class RuntimeOutput(StrictModel):
    """Identity of the emitted artifact and optional metric summary."""

    artifact: str = Field(min_length=1, max_length=256)
    schema_version: str = Field(min_length=1, max_length=128)
    payload_sha256: str = Field(pattern=SHA256.pattern)
    report_id: str | None = Field(default=None, pattern=RECORD_ID_PATTERN)
    metrics: dict[str, float] = Field(default_factory=dict)

    @field_validator("metrics", mode="after")
    @classmethod
    def freeze_metrics(cls, value: dict[str, float]) -> dict[str, float]:
        if any(not key.strip() for key in value):
            raise ValueError("output metric names must be non-empty")
        if any(not math.isfinite(item) for item in value.values()):
            raise ValueError("output metrics must be finite")
        return cast(dict[str, float], freeze_value(value))

    @field_validator("metrics", mode="before")
    @classmethod
    def metrics_must_be_numeric(cls, value: object) -> object:
        if not isinstance(value, dict):
            raise ValueError("output metrics must be an object")  # noqa: TRY004
        for key, item in value.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError("output metric names must be non-empty")
            _strict_number(item, f"output metric {key}")
        return value

    @model_validator(mode="after")
    def validate_hash(self) -> RuntimeOutput:
        if self.payload_sha256 == "0" * 64:
            raise ValueError("output payload hash cannot be an all-zero placeholder")
        return self


class RuntimeStage(StrictModel):
    """A separately measured pipeline phase.

    Unavailable phases are represented explicitly with ``measured=false`` and
    a note instead of inventing a zero-duration measurement.
    """

    name: str = Field(min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_.-]*$")
    measured: bool
    duration_ms: float | None = Field(default=None, ge=0.0)
    items: int | None = Field(default=None, ge=0, le=10_000_000)
    throughput_per_s: float | None = Field(default=None, ge=0.0)
    note: str = Field(default="", max_length=512)

    @field_validator("measured", mode="before")
    @classmethod
    def measured_must_be_boolean(cls, value: object) -> bool:
        return validate_bool(value)

    @field_validator("duration_ms", "throughput_per_s", mode="after")
    @classmethod
    def timing_must_be_finite(cls, value: float | None) -> float | None:
        return None if value is None else _finite(value, "timing value")

    @field_validator("duration_ms", "throughput_per_s", mode="before")
    @classmethod
    def timing_must_be_numeric(cls, value: object, info: Any) -> object:
        return None if value is None else _strict_number(value, info.field_name)

    @field_validator("items", mode="before")
    @classmethod
    def items_must_be_integer(cls, value: object) -> object:
        return None if value is None else _strict_integer(value, "items")

    @model_validator(mode="after")
    def validate_measurement(self) -> RuntimeStage:
        if self.measured:
            if self.duration_ms is None:
                raise ValueError("measured stage requires duration_ms")
            if self.items is not None and self.duration_ms > 0:
                expected = self.items / (self.duration_ms / 1000.0)
                if self.throughput_per_s is None or not math.isclose(
                    self.throughput_per_s, expected, rel_tol=2e-4, abs_tol=1e-6
                ):
                    raise ValueError("stage throughput must match duration and item count")
            elif self.throughput_per_s is not None:
                raise ValueError("throughput requires positive duration and item count")
        elif any(
            value is not None for value in (self.duration_ms, self.items, self.throughput_per_s)
        ):
            raise ValueError("unmeasured stage cannot contain timing values")
        if not self.measured and not self.note.strip():
            raise ValueError("unmeasured stage requires an explanatory note")
        return self


class RuntimeAuditRecord(StrictModel):
    """Signed-by-hash runtime audit record.

    ``benchmark_claim_available`` is intentionally constrained to ``False``
    for this milestone.  A future authorized evaluator may define a new schema
    once dataset, model, and evaluator provenance are complete.
    """

    schema_version: Literal["roadsense.runtime-audit/v1"]
    record_id: str = Field(pattern=RECORD_ID_PATTERN)
    run_mode: Literal["fixture_dry_run", "local_experiment"]
    evidence_level: EvidenceLevel
    evaluation_authorized: bool
    frozen: bool
    benchmark_claim_available: Literal[False]
    started_at_utc: datetime
    device: RuntimeDevice
    dependencies: dict[str, str]
    input: RuntimeInput
    output: RuntimeOutput
    input_sha256: str = Field(pattern=SHA256.pattern)
    output_sha256: str = Field(pattern=SHA256.pattern)
    iterations: int = Field(ge=1, le=10_000)
    wall_time_ms: float = Field(ge=0.0)
    throughput_fps: float | None = Field(default=None, ge=0.0)
    stages: tuple[RuntimeStage, ...] = Field(min_length=1)
    claim_boundary: str = Field(min_length=1, max_length=2_000)

    @field_validator("evaluation_authorized", "frozen", mode="before")
    @classmethod
    def flags_must_be_boolean(cls, value: object) -> bool:
        return validate_bool(value)

    @field_validator("dependencies", mode="after")
    @classmethod
    def freeze_dependencies(cls, value: dict[str, str]) -> dict[str, str]:
        if not value:
            raise ValueError("runtime record requires dependency versions")
        for package, version in value.items():
            if not package.strip() or not version.strip():
                raise ValueError("dependency names and versions must be non-empty")
        return cast(dict[str, str], freeze_value(value))

    @field_validator("dependencies", mode="before")
    @classmethod
    def dependencies_must_be_strings(cls, value: object) -> object:
        if not isinstance(value, dict):
            raise ValueError("dependencies must be an object")  # noqa: TRY004
        for package, version in value.items():
            if not isinstance(package, str) or not isinstance(version, str):
                raise ValueError("dependency names and versions must be strings")  # noqa: TRY004
        return value

    @field_validator("wall_time_ms", "throughput_fps", mode="after")
    @classmethod
    def top_level_timing_must_be_finite(cls, value: float | None) -> float | None:
        return None if value is None else _finite(value, "runtime timing")

    @field_validator("wall_time_ms", "throughput_fps", mode="before")
    @classmethod
    def top_level_timing_must_be_numeric(cls, value: object, info: Any) -> object:
        return None if value is None else _strict_number(value, info.field_name)

    @field_validator("iterations", mode="before")
    @classmethod
    def iterations_must_be_integer(cls, value: object) -> int:
        return _strict_integer(value, "iterations")

    @field_validator("started_at_utc", mode="after")
    @classmethod
    def timestamp_must_be_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() != timezone.utc.utcoffset(value):
            raise ValueError("started_at_utc must be timezone-aware UTC")
        return value

    @model_validator(mode="after")
    def validate_record(self) -> RuntimeAuditRecord:
        if self.input_sha256 != self.input.payload_sha256:
            raise ValueError("input_sha256 must match input.payload_sha256")
        if self.output_sha256 != self.output.payload_sha256:
            raise ValueError("output_sha256 must match output.payload_sha256")
        if self.iterations != self.input.iterations:
            raise ValueError("iterations must match input.iterations")
        if self.run_mode == "fixture_dry_run":
            if self.evidence_level is not EvidenceLevel.FIXTURE:
                raise ValueError("fixture dry-runs must use fixture evidence")
            if self.input.source != "deterministic_geometric_fixture":
                raise ValueError("fixture dry-runs must identify the deterministic fixture source")
        elif self.evidence_level is EvidenceLevel.FIXTURE:
            raise ValueError("local experiments cannot be labelled fixture evidence")
        if self.evidence_level is EvidenceLevel.FIXTURE and (
            self.evaluation_authorized or self.frozen
        ):
            raise ValueError("fixture runtime records cannot be authorized or frozen")
        if self.benchmark_claim_available is not False:
            raise ValueError("runtime benchmark claims are not available in this milestone")
        names = [stage.name for stage in self.stages]
        if len(names) != len(set(names)):
            raise ValueError("runtime stages must be unique")
        if self.throughput_fps is not None:
            if self.wall_time_ms <= 0:
                raise ValueError("throughput requires positive wall_time_ms")
            expected = self.input.frame_count * self.iterations / (self.wall_time_ms / 1000.0)
            if not math.isclose(self.throughput_fps, expected, rel_tol=2e-4, abs_tol=1e-6):
                raise ValueError("throughput_fps must match wall_time_ms and frame count")
        payload = self.model_dump(mode="json")
        if self.record_id != compute_runtime_record_id(payload):
            raise ValueError("record_id does not bind the complete runtime payload")
        return self


def compute_runtime_record_id(payload: Mapping[str, Any]) -> str:
    """Return the short canonical identity for a complete runtime record."""

    material = {key: value for key, value in payload.items() if key != "record_id"}
    # Pydantic emits UTC datetimes with a ``Z`` suffix while callers may supply
    # the equivalent ``+00:00`` spelling.  Normalize both forms before hashing
    # so the identity is representation-independent.
    timestamp = material.get("started_at_utc")
    if isinstance(timestamp, str) and timestamp.endswith("+00:00"):
        material["started_at_utc"] = f"{timestamp[:-6]}Z"
    return canonical_sha256(material)[:16]


def _dependency_versions() -> dict[str, str]:
    # Read optional dependency metadata from the environment, but take the
    # project version from the imported source package.  In an editable
    # checkout, a stale ``*.dist-info`` directory can otherwise make an audit
    # claim that the running source is an older release.
    names = ("numpy", "pydantic", "fastapi", "uvicorn")
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    versions["roadsense-perception"] = __version__
    return dict(sorted(versions.items()))


def collect_runtime_device() -> RuntimeDevice:
    """Collect reproducibility metadata without exposing host identifiers."""

    return RuntimeDevice(
        device=os.getenv("ROADSENSE_DEVICE", "cpu").strip() or "cpu",
        operating_system=platform.system() or "unknown",
        operating_system_version=platform.release() or "unknown",
        machine=platform.machine() or "unknown",
        processor=platform.processor() or "unknown",
        python_implementation=platform.python_implementation() or "unknown",
        python_version=platform.python_version()
        or f"{sys.version_info.major}.{sys.version_info.minor}",
        cpu_count=os.cpu_count(),
    )


def _stage(name: str, duration_ns: int, items: int, note: str) -> RuntimeStage:
    duration_ms = max(0.0, duration_ns / 1_000_000.0)
    throughput = None
    if duration_ms > 0 and items > 0:
        throughput = items / (duration_ms / 1000.0)
    return RuntimeStage(
        name=name,
        measured=True,
        duration_ms=duration_ms,
        items=items,
        throughput_per_s=throughput,
        note=note,
    )


def _fixture_input_material(bundle: FixtureBundle, *, iterations: int) -> dict[str, object]:
    """Build a compact, content-bound identity for the consumed fixture bundle.

    Frames are represented as canonical JSON while dense masks are represented
    by dtype/shape plus a binary SHA-256.  This keeps the audit record small
    without reducing the input hash to a hand-written fixture name.
    """

    def array_identity(value: Any) -> dict[str, object]:
        array = np.ascontiguousarray(value)
        return {
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
        }

    return {
        "fixture_id": "roadsense-city-loop-v1",
        "iterations": iterations,
        "truth_frames": [frame.model_dump(mode="json") for frame in bundle.truth_frames],
        "prediction_frames": [frame.model_dump(mode="json") for frame in bundle.prediction_frames],
        "truth_masks": array_identity(bundle.truth_masks),
        "prediction_masks": array_identity(bundle.prediction_masks),
    }


def build_fixture_runtime_record(*, iterations: int = 1) -> RuntimeAuditRecord:
    """Run the deterministic fixture and return an auditable timing record.

    ``iterations`` repeats the complete synthetic pipeline.  Timings are useful
    for local regression diagnostics only; they must not be reported as model
    latency or FPS because this path performs no learned inference or browser
    rendering.
    """

    if (
        isinstance(iterations, bool)
        or not isinstance(iterations, int)
        or not 1 <= iterations <= 10_000
    ):
        raise ValueError("iterations must be an integer in [1, 10000]")
    started = datetime.now(timezone.utc)
    wall_start = time.perf_counter_ns()
    generation_ns = 0
    evaluation_ns = 0
    serialization_ns = 0
    demo: dict[str, object] | None = None
    metrics: dict[str, object] | None = None
    for _ in range(iterations):
        phase_start = time.perf_counter_ns()
        bundle = build_fixture_bundle()
        generation_ns += time.perf_counter_ns() - phase_start

        phase_start = time.perf_counter_ns()
        metrics = build_fixture_metrics(bundle)
        evaluation_ns += time.perf_counter_ns() - phase_start

        phase_start = time.perf_counter_ns()
        demo = build_demo_payload(bundle)
        serialization_ns += time.perf_counter_ns() - phase_start
    if demo is None or metrics is None:  # defensive; loop is non-empty by contract
        raise RuntimeError("fixture runtime produced no output")
    wall_time_ms = max(0.0, (time.perf_counter_ns() - wall_start) / 1_000_000.0)
    frame_count = FRAME_COUNT
    input_payload = _fixture_input_material(bundle, iterations=iterations)
    input_sha256 = canonical_sha256(input_payload)
    input_material = {
        "source": "deterministic_geometric_fixture",
        "fixture_id": "roadsense-city-loop-v1",
        "frame_count": frame_count,
        "iterations": iterations,
        "payload_sha256": input_sha256,
    }
    output_material = {"demo": demo, "metrics": metrics}
    output_sha256 = canonical_sha256(output_material)
    flattened_metrics = {
        "detection_ap50": float(cast(dict[str, Any], metrics["detection"])["ap"]),
        "segmentation_mean_iou": float(cast(dict[str, Any], metrics["segmentation"])["mean_iou"]),
        "tracking_mota": float(cast(dict[str, Any], metrics["tracking"])["mota"]),
    }
    stages = (
        _stage(
            "fixture_generation",
            generation_ns,
            frame_count * iterations,
            "Synthetic geometry and deterministic tracker update; no model inference.",
        ),
        _stage(
            "evaluation",
            evaluation_ns,
            frame_count * iterations,
            "RoadSense fixture metric protocols only; not a public-dataset evaluator.",
        ),
        _stage(
            "serialization",
            serialization_ns,
            frame_count * iterations,
            "Build the API/Pages replay payload in memory, including its fixture metric summary.",
        ),
        RuntimeStage(
            name="inference",
            measured=False,
            note="Unavailable: fixture dry-run does not load a learned model.",
        ),
        RuntimeStage(
            name="rendering",
            measured=False,
            note="Unavailable: browser rendering is outside the Python runner.",
        ),
    )
    throughput_fps = None
    if wall_time_ms > 0:
        throughput_fps = frame_count * iterations / (wall_time_ms / 1000.0)
    raw: dict[str, Any] = {
        "schema_version": RUNTIME_SCHEMA,
        "record_id": "0" * 16,
        "run_mode": "fixture_dry_run",
        "evidence_level": EvidenceLevel.FIXTURE.value,
        "evaluation_authorized": False,
        "frozen": False,
        "benchmark_claim_available": False,
        "started_at_utc": started.isoformat().replace("+00:00", "Z"),
        "device": collect_runtime_device().model_dump(mode="json"),
        "dependencies": _dependency_versions(),
        "input": {
            **input_material,
        },
        "output": {
            "artifact": "roadsense.demo.payload",
            "schema_version": "roadsense.demo/v1",
            "payload_sha256": output_sha256,
            "report_id": None,
            "metrics": flattened_metrics,
        },
        "input_sha256": input_sha256,
        "output_sha256": output_sha256,
        "iterations": iterations,
        "wall_time_ms": wall_time_ms,
        "throughput_fps": throughput_fps,
        "stages": [stage.model_dump(mode="json") for stage in stages],
        "claim_boundary": (
            "Fixture dry-run timings validate runtime instrumentation and payload plumbing only. "
            "They are not learned-model latency, FPS, throughput, robustness, or public-dataset evidence."
        ),
    }
    raw["record_id"] = compute_runtime_record_id(raw)
    return RuntimeAuditRecord.model_validate(raw)


def write_fixture_runtime_record(path: Path, *, iterations: int = 1) -> Path:
    """Generate and atomically write a fixture runtime record."""

    record = build_fixture_runtime_record(iterations=iterations)
    return write_json_atomic(path, record.model_dump(mode="json"))


__all__ = [
    "RUNTIME_SCHEMA",
    "RuntimeAuditRecord",
    "RuntimeDevice",
    "RuntimeInput",
    "RuntimeOutput",
    "RuntimeStage",
    "build_fixture_runtime_record",
    "collect_runtime_device",
    "compute_runtime_record_id",
    "write_fixture_runtime_record",
]
