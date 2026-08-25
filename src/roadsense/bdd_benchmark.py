"""Fail-closed BDD100K Detection 2020 benchmark receipts.

This module deliberately validates *provenance*, not model quality.  It never
downloads BDD100K, invokes an evaluator, or opens model/prediction files.  A
runner supplies content digests for those immutable inputs and outputs, then
uses :func:`build_bdd100k_detection_receipt` to produce a canonical receipt.

Only the public BDD100K Detection 2020 validation protocol is accepted.  The
schema has no path fields and rejects path-like identifiers, which keeps a
published receipt from disclosing local workspaces or artifact locations.
"""

from __future__ import annotations

import math
import numbers
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import Field, field_validator, model_validator

from roadsense.contracts import SHA256, StrictModel, freeze_value, validate_bool
from roadsense.json_io import canonical_sha256, load_strict_json

BDD100K_DETECTION_BENCHMARK_SCHEMA = "roadsense.bdd100k-detection-benchmark/v1"
BDD100K_DETECTION_SCOPE = "BDD100K Detection 2020 validation"
BDD100K_DEVKIT_ID = "bdd100k-devkit"
BDD100K_DEVKIT_REPOSITORY = "bdd100k/bdd100k"
BDD100K_DEVKIT_COMMIT = "9ac17c6c7c51d2fc83065fccd707cd5b1882a293"
BDD100K_OFFICIAL_IMAGE_COUNT = 10_000
BDD100K_OFFICIAL_IMAGES_MD5 = "5a0359c86a0b8713adab1eee9a3041cb"
BDD100K_OFFICIAL_LABELS_MD5 = "b86a3e1b7edbcad421b7dad2b3987c94"
BDD100K_REQUIRED_EVALUATOR_PACKAGES = {
    "bdd100k": "1.0.0",
    "motmetrics": "1.4.0",
    "numpy": "1.26.4",
    "pycocotools": "2.0.7",
    "pydantic": "1.10.15",
    "scalabel": "0.3.1",
}

_REPORT_ID_PATTERN = r"^[0-9a-f]{64}$"
_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9._-]{1,127}$")
_PACKAGE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_PACKAGE_VERSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.+!=<>~-]{0,127}$")
_METRIC_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")


class BDD100KBenchmarkReceiptError(ValueError):
    """Raised when a benchmark receipt cannot be safely loaded or built."""


def _strict_text(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")  # noqa: TRY004
    if not value or value != value.strip():
        raise ValueError(f"{field_name} must not be blank or have surrounding whitespace")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ValueError(f"{field_name} must not contain control characters")
    return value


def _reject_path_like_text(value: str, *, field_name: str) -> str:
    """Reject all user-controlled path and URL separators.

    The receipt contains hashes rather than paths.  Treating path-like text as
    invalid is safer than attempting to redact it after it has entered an
    immutable evidence payload.
    """

    if any(character in value for character in ("/", "\\", ":", "@")):
        raise ValueError(f"{field_name} must not expose local paths or URLs")
    return value


def _strict_identifier(value: object, *, field_name: str) -> str:
    text = _reject_path_like_text(_strict_text(value, field_name=field_name), field_name=field_name)
    if not _IDENTIFIER.fullmatch(text):
        raise ValueError(f"{field_name} must be a lowercase identifier")
    return text


def _strict_sha256(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 string")  # noqa: TRY004
    if not SHA256.fullmatch(value):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
    if value == "0" * 64:
        raise ValueError(f"{field_name} cannot be an all-zero placeholder")
    return value


def _strict_metric_mapping(value: object) -> dict[str, float]:
    if not isinstance(value, dict):
        raise ValueError("metrics must be an object of finite numbers")  # noqa: TRY004
    if not value:
        raise ValueError("metrics must not be empty")
    if len(value) > 512:
        raise ValueError("metrics contains too many values")

    result: dict[str, float] = {}
    for name, metric in value.items():
        if not isinstance(name, str) or not _METRIC_NAME.fullmatch(name):
            raise ValueError("metric names must be safe non-empty identifiers")
        if isinstance(metric, bool) or not isinstance(metric, numbers.Real):
            raise ValueError(f"metrics.{name} must be a number")  # noqa: TRY004
        normalized = float(metric)
        if not math.isfinite(normalized):
            raise ValueError(f"metrics.{name} must be finite")
        result[name] = normalized
    return result


def _strict_package_mapping(value: object) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ValueError("dependency packages must be an object")  # noqa: TRY004
    if not value:
        raise ValueError("dependency packages must not be empty")
    if len(value) > 512:
        raise ValueError("dependency packages contains too many values")

    result: dict[str, str] = {}
    for name, version in value.items():
        if not isinstance(name, str) or not _PACKAGE_NAME.fullmatch(name):
            raise ValueError("dependency package names must be safe identifiers")
        text = _reject_path_like_text(
            _strict_text(version, field_name=f"dependency version for {name}"),
            field_name=f"dependency version for {name}",
        )
        if not _PACKAGE_VERSION.fullmatch(text):
            raise ValueError("dependency package versions contain unsupported characters")
        result[name] = text
    return result


class BDD100KDatasetProvenance(StrictModel):
    """Hash-bound public BDD100K Detection validation inputs."""

    dataset: Literal["BDD100K"]
    task: Literal["detection"]
    release: Literal["2020"]
    split: Literal["val"]
    image_count: Literal[10_000]
    images_package_md5: Literal["5a0359c86a0b8713adab1eee9a3041cb"]
    labels_package_md5: Literal["b86a3e1b7edbcad421b7dad2b3987c94"]
    archive_sha256: str = Field(pattern=SHA256.pattern)
    tree_sha256: str = Field(pattern=SHA256.pattern)
    ground_truth_sha256: str = Field(pattern=SHA256.pattern)
    split_manifest_sha256: str = Field(pattern=SHA256.pattern)

    @field_validator(
        "archive_sha256",
        "tree_sha256",
        "ground_truth_sha256",
        "split_manifest_sha256",
        mode="before",
    )
    @classmethod
    def hashes_must_be_real_sha256(cls, value: object, info: Any) -> str:
        return _strict_sha256(value, field_name=str(info.field_name))


class BDD100KModelProvenance(StrictModel):
    """Model and BDD100K ontology translation evidence."""

    model_id: str = Field(min_length=2, max_length=128, pattern=_IDENTIFIER.pattern)
    artifact_sha256: str = Field(pattern=SHA256.pattern)
    manifest_sha256: str = Field(pattern=SHA256.pattern)
    ontology_map_sha256: str = Field(pattern=SHA256.pattern)

    @field_validator("model_id", mode="before")
    @classmethod
    def model_id_must_be_safe(cls, value: object) -> str:
        return _strict_identifier(value, field_name="model_id")

    @field_validator("artifact_sha256", "manifest_sha256", "ontology_map_sha256", mode="before")
    @classmethod
    def hashes_must_be_real_sha256(cls, value: object, info: Any) -> str:
        return _strict_sha256(value, field_name=str(info.field_name))


class BDD100KInferenceProvenance(StrictModel):
    """Hashes for the deterministic inference configuration and predictions."""

    config_sha256: str = Field(pattern=SHA256.pattern)
    prediction_sha256: str = Field(pattern=SHA256.pattern)

    @field_validator("config_sha256", "prediction_sha256", mode="before")
    @classmethod
    def hashes_must_be_real_sha256(cls, value: object, info: Any) -> str:
        return _strict_sha256(value, field_name=str(info.field_name))


class BDD100KDependencyProvenance(StrictModel):
    """Pinned evaluator environment without machine-local paths."""

    lock_sha256: str = Field(pattern=SHA256.pattern)
    packages: dict[str, str]

    @field_validator("lock_sha256", mode="before")
    @classmethod
    def lock_hash_must_be_real_sha256(cls, value: object) -> str:
        return _strict_sha256(value, field_name="lock_sha256")

    @field_validator("packages", mode="before")
    @classmethod
    def packages_must_be_safe(cls, value: object) -> dict[str, str]:
        return _strict_package_mapping(value)

    @field_validator("packages", mode="after")
    @classmethod
    def freeze_packages(cls, value: dict[str, str]) -> dict[str, str]:
        if value != BDD100K_REQUIRED_EVALUATOR_PACKAGES:
            raise ValueError(
                "dependency packages must exactly match the validated BDD100K evaluator lock"
            )
        return cast(dict[str, str], freeze_value(dict(sorted(value.items()))))


class BDD100KEvaluatorProvenance(StrictModel):
    """The only evaluator implementation accepted by this receipt version."""

    evaluator_id: Literal["bdd100k-devkit"]
    repository: Literal["bdd100k/bdd100k"]
    commit: Literal["9ac17c6c7c51d2fc83065fccd707cd5b1882a293"]
    config_sha256: str = Field(pattern=SHA256.pattern)
    dependencies: BDD100KDependencyProvenance

    @field_validator("config_sha256", mode="before")
    @classmethod
    def config_hash_must_be_real_sha256(cls, value: object) -> str:
        return _strict_sha256(value, field_name="evaluator config_sha256")


class BDD100KEvaluatorRun(StrictModel):
    """One independently recorded invocation of the pinned evaluator."""

    role: Literal["independent_a", "independent_b"]
    run_id: str = Field(min_length=2, max_length=128, pattern=_IDENTIFIER.pattern)
    output_sha256: str = Field(pattern=SHA256.pattern)
    metrics: dict[str, float]

    @field_validator("run_id", mode="before")
    @classmethod
    def run_id_must_be_safe(cls, value: object) -> str:
        return _strict_identifier(value, field_name="evaluator run_id")

    @field_validator("output_sha256", mode="before")
    @classmethod
    def output_hash_must_be_real_sha256(cls, value: object) -> str:
        return _strict_sha256(value, field_name="evaluator output_sha256")

    @field_validator("metrics", mode="before")
    @classmethod
    def metrics_must_be_finite_numbers(cls, value: object) -> dict[str, float]:
        return _strict_metric_mapping(value)

    @field_validator("metrics", mode="after")
    @classmethod
    def freeze_metrics(cls, value: dict[str, float]) -> dict[str, float]:
        return cast(dict[str, float], freeze_value(dict(sorted(value.items()))))


class BDD100KDetectionReceiptMaterial(StrictModel):
    """All benchmark receipt content except its self-binding report identifier."""

    schema_version: Literal["roadsense.bdd100k-detection-benchmark/v1"]
    scope: Literal["BDD100K Detection 2020 validation"]
    evidence_level: Literal["frozen_evaluation"]
    evaluation_authorized: Literal[True]
    frozen: Literal[True]
    benchmark_claim_available: Literal[True]
    dataset: BDD100KDatasetProvenance
    model: BDD100KModelProvenance
    inference: BDD100KInferenceProvenance
    evaluator: BDD100KEvaluatorProvenance
    evaluator_runs: tuple[BDD100KEvaluatorRun, BDD100KEvaluatorRun]

    @field_validator("evaluation_authorized", "frozen", "benchmark_claim_available", mode="before")
    @classmethod
    def benchmark_flags_must_be_boolean(cls, value: object) -> bool:
        return validate_bool(value)

    @model_validator(mode="after")
    def validate_independent_evaluator_runs(self) -> BDD100KDetectionReceiptMaterial:
        first, second = self.evaluator_runs
        if (first.role, second.role) != ("independent_a", "independent_b"):
            raise ValueError(
                "evaluator_runs must be canonically ordered as independent_a then independent_b"
            )
        if first.run_id == second.run_id:
            raise ValueError("independent evaluator runs must have distinct run_id values")
        if first.metrics != second.metrics:
            raise ValueError("independent evaluator runs must report identical metrics")
        return self


class BDD100KDetectionBenchmarkReceipt(BDD100KDetectionReceiptMaterial):
    """Immutable, canonical BDD100K Detection benchmark claim receipt."""

    report_id: str = Field(pattern=_REPORT_ID_PATTERN)

    @field_validator("report_id", mode="before")
    @classmethod
    def report_id_must_be_full_sha256(cls, value: object) -> str:
        return _strict_sha256(value, field_name="report_id")

    @model_validator(mode="after")
    def validate_report_identity(self) -> BDD100KDetectionBenchmarkReceipt:
        payload = self.model_dump(mode="json")
        if self.report_id != compute_bdd100k_detection_report_id(payload):
            raise ValueError("report_id does not bind the complete benchmark receipt")
        return self


def compute_bdd100k_detection_report_id(payload: Mapping[str, Any]) -> str:
    """Return the full canonical SHA-256 identity for a receipt payload.

    The identifier itself is omitted before hashing.  Unlike the fixture
    report's display-oriented short ID, formal benchmark evidence keeps the
    full digest to make collision resistance explicit.
    """

    material = {key: value for key, value in payload.items() if key != "report_id"}
    return canonical_sha256(material)


def build_bdd100k_detection_receipt(
    payload: Mapping[str, Any],
) -> BDD100KDetectionBenchmarkReceipt:
    """Normalize, bind, and validate a runner-supplied benchmark payload.

    ``payload`` may omit ``report_id``.  If it includes one, it must already
    equal the canonical identifier; the builder never silently replaces a
    stale or tampered identity.
    """

    if not isinstance(payload, Mapping):
        raise BDD100KBenchmarkReceiptError("benchmark receipt payload must be an object")
    raw = dict(payload)
    has_report_id = "report_id" in raw
    supplied_report_id = raw.pop("report_id", None)
    material = BDD100KDetectionReceiptMaterial.model_validate(raw)
    normalized = material.model_dump(mode="json")
    report_id = compute_bdd100k_detection_report_id(normalized)
    if has_report_id and supplied_report_id != report_id:
        raise BDD100KBenchmarkReceiptError("report_id does not bind the complete benchmark receipt")
    normalized["report_id"] = report_id
    return BDD100KDetectionBenchmarkReceipt.model_validate(normalized)


def validate_bdd100k_detection_receipt(value: object) -> BDD100KDetectionBenchmarkReceipt:
    """Validate a complete receipt, including the canonical report identity."""

    return BDD100KDetectionBenchmarkReceipt.model_validate(value)


def load_bdd100k_detection_receipt(path: Path | str) -> BDD100KDetectionBenchmarkReceipt:
    """Strictly load a JSON receipt and reject duplicate keys or non-finite JSON."""

    try:
        payload = load_strict_json(Path(path))
    except (OSError, TypeError, UnicodeError, ValueError) as exc:
        raise BDD100KBenchmarkReceiptError(
            f"unable to load BDD100K benchmark receipt: {path}"
        ) from exc
    return validate_bdd100k_detection_receipt(payload)


__all__ = [
    "BDD100K_DETECTION_BENCHMARK_SCHEMA",
    "BDD100K_DETECTION_SCOPE",
    "BDD100K_DEVKIT_COMMIT",
    "BDD100K_DEVKIT_ID",
    "BDD100K_DEVKIT_REPOSITORY",
    "BDD100K_OFFICIAL_IMAGES_MD5",
    "BDD100K_OFFICIAL_IMAGE_COUNT",
    "BDD100K_OFFICIAL_LABELS_MD5",
    "BDD100K_REQUIRED_EVALUATOR_PACKAGES",
    "BDD100KBenchmarkReceiptError",
    "BDD100KDatasetProvenance",
    "BDD100KDependencyProvenance",
    "BDD100KDetectionBenchmarkReceipt",
    "BDD100KDetectionReceiptMaterial",
    "BDD100KEvaluatorProvenance",
    "BDD100KEvaluatorRun",
    "BDD100KInferenceProvenance",
    "BDD100KModelProvenance",
    "build_bdd100k_detection_receipt",
    "compute_bdd100k_detection_report_id",
    "load_bdd100k_detection_receipt",
    "validate_bdd100k_detection_receipt",
]
