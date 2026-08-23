"""Explicit model-artifact contracts and adapter registration.

The public demo intentionally does not load a model. This module provides the
dependency-light seam used by a future local inference runner: a model artifact
is described by a frozen manifest, the checkpoint is verified against that
manifest, and an adapter can then be registered under a stable name.

Nothing here imports a model framework or downloads a checkpoint. Verification
is an explicit operation and only accepts files below an operator-provided
local root, keeping the Pages/fixture path independent from optional runtimes.
"""

from __future__ import annotations

import hashlib
import math
import re
import threading
from collections.abc import Iterator
from pathlib import Path, PurePosixPath
from typing import Any, Generic, Protocol, TypeVar, cast, runtime_checkable

from pydantic import Field, field_validator, model_validator

from roadsense.contracts import SHA256, StrictModel, TaskKind, freeze_value, validate_bool
from roadsense.json_io import canonical_sha256, load_strict_json

MODEL_ARTIFACT_SCHEMA = "roadsense.model-artifact-manifest/v1"
_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9._-]{1,127}$")
_VERSION = re.compile(r"^[^\s]{1,128}$")
_PATH_DRIVE = re.compile(r"^[A-Za-z]:")


def _validate_metadata(value: object, *, field_name: str, depth: int = 0) -> object:
    """Validate small JSON metadata blocks without accepting NaN or huge trees."""

    if depth > 16:
        raise ValueError(f"{field_name} metadata nesting is too deep")
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{field_name} metadata must contain finite numbers")
        return value
    if isinstance(value, list | tuple):
        if len(value) > 256:
            raise ValueError(f"{field_name} metadata arrays are too large")
        return tuple(
            _validate_metadata(item, field_name=field_name, depth=depth + 1) for item in value
        )
    if isinstance(value, dict):
        if len(value) > 256:
            raise ValueError(f"{field_name} metadata objects are too large")
        result: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key.strip() or key != key.strip() or len(key) > 128:
                raise ValueError(f"{field_name} metadata keys must be non-empty strings")
            result[key] = _validate_metadata(item, field_name=field_name, depth=depth + 1)
        return result
    raise ValueError(f"{field_name} metadata must be JSON-compatible")


def _reject_boolean_integer(value: object, *, field_name: str) -> object:
    """Prevent Python's ``bool``-is-an-``int`` coercion in numeric fields."""

    if isinstance(value, bool):
        raise ValueError(  # noqa: TRY004 - Pydantic must wrap this in ValidationError.
            f"{field_name} must be an integer, not a boolean"
        )
    return value


class ArtifactVerificationError(ValueError):
    """Raised when a local model artifact cannot be verified fail-closed."""


class OntologyClass(StrictModel):
    """One stable model output class."""

    category_id: int = Field(ge=0, le=1_000_000)
    label: str = Field(min_length=1, max_length=128)

    @field_validator("category_id", mode="before")
    @classmethod
    def category_id_must_be_integer(cls, value: object) -> object:
        return _reject_boolean_integer(value, field_name="category_id")

    @field_validator("label")
    @classmethod
    def label_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("ontology labels must not be blank")
        if value != value.strip():
            raise ValueError("ontology labels must not have surrounding whitespace")
        return value

    @property
    def id(self) -> int:
        return self.category_id


class ArtifactInputSpec(StrictModel):
    """Input tensor/image assumptions frozen with the checkpoint."""

    width: int = Field(ge=1, le=100_000)
    height: int = Field(ge=1, le=100_000)
    channels: int = Field(ge=1, le=16)
    dtype: str = Field(min_length=1, max_length=32, pattern=r"^[A-Za-z0-9_.-]+$")
    layout: str = Field(min_length=1, max_length=16, pattern=r"^[A-Za-z0-9]+$")
    color_order: str = Field(min_length=1, max_length=16, pattern=r"^[A-Za-z0-9_-]+$")
    coordinate_space: str = Field(min_length=1, max_length=64, pattern=r"^[a-z0-9_.-]+$")

    @field_validator("width", "height", "channels", mode="before")
    @classmethod
    def dimensions_must_be_integers(cls, value: object, info: Any) -> object:
        return _reject_boolean_integer(value, field_name=str(info.field_name))


class ArtifactOutputSpec(StrictModel):
    """Output schema and coordinate convention expected from an adapter."""

    output_schema: str = Field(min_length=1, max_length=128, pattern=r"^[a-z0-9][a-z0-9_.\-/]*$")
    coordinate_space: str = Field(min_length=1, max_length=64, pattern=r"^[a-z0-9_.-]+$")
    score_semantics: str = Field(min_length=1, max_length=128)
    output_shape: tuple[int | None, ...] | None = None

    @field_validator("output_shape", mode="before")
    @classmethod
    def output_shape_dimensions_must_be_integers(cls, value: object) -> object:
        if value is not None and isinstance(value, (list, tuple)):
            for dimension in value:
                if dimension is not None:
                    _reject_boolean_integer(dimension, field_name="output_shape dimension")
        return value

    @field_validator("score_semantics")
    @classmethod
    def score_semantics_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("score_semantics must not be blank")
        return value

    @field_validator("output_shape", mode="after")
    @classmethod
    def output_shape_must_be_bounded(
        cls, value: tuple[int | None, ...] | None
    ) -> tuple[int | None, ...] | None:
        if value is not None:
            if not value or len(value) > 16:
                raise ValueError("output_shape must contain between 1 and 16 dimensions")
            if any(dimension is not None and dimension <= 0 for dimension in value):
                raise ValueError("output_shape dimensions must be positive or null")
        return value


class ArtifactPreprocessingSpec(StrictModel):
    """Deterministic preprocessing values needed to reproduce an evaluation."""

    resize_mode: str = Field(min_length=1, max_length=32, pattern=r"^[a-z0-9_.-]+$")
    scale: float = Field(gt=0.0, le=1_000.0)
    mean: tuple[float, ...] = ()
    std: tuple[float, ...] = ()
    pad_value: float = 0.0

    @field_validator("scale", "pad_value", mode="after")
    @classmethod
    def preprocessing_values_must_be_finite(cls, value: float) -> float:
        import math

        if not math.isfinite(value):
            raise ValueError("preprocessing values must be finite")
        return value

    @field_validator("mean", "std", mode="after")
    @classmethod
    def normalization_values_must_be_finite(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        import math

        if any(not math.isfinite(item) for item in value):
            raise ValueError("normalization values must be finite")
        return value

    @model_validator(mode="after")
    def validate_normalization(self) -> ArtifactPreprocessingSpec:
        if bool(self.mean) != bool(self.std):
            raise ValueError("mean and std must be supplied together")
        if self.mean and len(self.mean) != len(self.std):
            raise ValueError("mean and std must have equal lengths")
        if any(item <= 0 for item in self.std):
            raise ValueError("normalization std values must be positive")
        return self


class ArtifactRuntimeSpec(StrictModel):
    """Runtime assumptions captured without importing a runtime package."""

    device: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_.:/-]+$")
    precision: str = Field(min_length=1, max_length=32, pattern=r"^[A-Za-z0-9_.-]+$")
    runtime_name: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_.-]+$")
    runtime_version: str = Field(min_length=1, max_length=128, pattern=r"^[^\s]+$")
    opset: int | None = Field(default=None, ge=1, le=1_000)
    deterministic: bool = False

    @field_validator("opset", mode="before")
    @classmethod
    def opset_must_be_integer(cls, value: object) -> object:
        return _reject_boolean_integer(value, field_name="opset")

    @field_validator("deterministic", mode="before")
    @classmethod
    def deterministic_must_be_boolean(cls, value: object) -> bool:
        return validate_bool(value)


class ModelArtifactManifest(StrictModel):
    """Complete, hash-bound description of a locally stored model artifact."""

    schema_version: str = MODEL_ARTIFACT_SCHEMA
    artifact_id: str = Field(min_length=2, max_length=128, pattern=_IDENTIFIER.pattern)
    artifact_path: str = Field(min_length=1, max_length=4_096)
    artifact_sha256: str = Field(pattern=SHA256.pattern)
    artifact_size_bytes: int | None = Field(default=None, ge=1)
    artifact_format: str = Field(min_length=1, max_length=32, pattern=r"^[a-z0-9][a-z0-9_.+-]*$")
    framework: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_.+-]+$")
    framework_version: str = Field(min_length=1, max_length=128, pattern=_VERSION.pattern)
    backend: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_.+-]+$")
    backend_version: str = Field(min_length=1, max_length=128, pattern=_VERSION.pattern)
    tasks: tuple[TaskKind, ...] = Field(min_length=1)
    ontology: tuple[OntologyClass, ...] = Field(min_length=1)
    input: ArtifactInputSpec
    output: ArtifactOutputSpec
    preprocessing: ArtifactPreprocessingSpec
    runtime: ArtifactRuntimeSpec
    license_id: str = Field(min_length=1, max_length=256)
    source: str = Field(min_length=1, max_length=2_048)
    dependency_lock_path: str | None = Field(default=None, max_length=4_096)
    dependency_lock_sha256: str | None = Field(default=None, pattern=SHA256.pattern)
    graph_metadata: dict[str, Any] = Field(default_factory=dict, validate_default=True)
    calibration: dict[str, Any] | None = None
    quantization: dict[str, Any] | None = None
    notes: str = Field(default="", max_length=2_000)

    @field_validator("artifact_size_bytes", mode="before")
    @classmethod
    def artifact_size_must_be_integer(cls, value: object) -> object:
        if value is None:
            return value
        return _reject_boolean_integer(value, field_name="artifact_size_bytes")

    @field_validator("schema_version")
    @classmethod
    def schema_must_be_supported(cls, value: str) -> str:
        if value != MODEL_ARTIFACT_SCHEMA:
            raise ValueError(f"unsupported model artifact schema: {value}")
        return value

    @field_validator("artifact_path", "dependency_lock_path")
    @classmethod
    def paths_must_be_safe_relative_paths(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.replace("\\", "/")
        if normalized != normalized.strip():
            raise ValueError("artifact paths must not have surrounding whitespace")
        if any(ord(character) < 32 or ord(character) == 127 for character in normalized):
            raise ValueError("artifact paths must not contain control characters")
        if normalized.startswith("/") or _PATH_DRIVE.match(normalized):
            raise ValueError("artifact paths must be relative to the supplied artifact root")
        parts = PurePosixPath(normalized).parts
        if not parts or any(part in {"", ".", ".."} for part in parts):
            raise ValueError("artifact paths must be normalized relative paths")
        if ":" in parts[0]:
            raise ValueError("artifact paths must not contain a drive or URI scheme")
        canonical = "/".join(parts)
        if normalized != canonical:
            raise ValueError("artifact paths must be normalized relative paths")
        return canonical

    @field_validator("artifact_sha256", "dependency_lock_sha256")
    @classmethod
    def hashes_must_not_be_placeholders(cls, value: str | None) -> str | None:
        if value is not None and value == "0" * 64:
            raise ValueError("artifact hashes cannot be all-zero placeholders")
        return value

    @field_validator("source", "license_id")
    @classmethod
    def text_fields_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text fields must not be blank")
        return value

    @field_validator("graph_metadata", "calibration", "quantization", mode="after")
    @classmethod
    def metadata_must_be_finite_json(
        cls, value: dict[str, Any] | None, info: Any
    ) -> dict[str, Any] | None:
        if value is None:
            return None
        field_name = str(info.field_name)
        checked = _validate_metadata(value, field_name=field_name)
        return cast(dict[str, Any], freeze_value(checked))

    @model_validator(mode="after")
    def validate_manifest(self) -> ModelArtifactManifest:
        task_values = tuple(self.tasks)
        if len(set(task_values)) != len(task_values):
            raise ValueError("tasks must be non-empty and unique")
        ontology_ids = tuple(item.category_id for item in self.ontology)
        if len(set(ontology_ids)) != len(ontology_ids):
            raise ValueError("ontology category IDs must be unique")
        labels = tuple(item.label.casefold() for item in self.ontology)
        if len(set(labels)) != len(labels):
            raise ValueError("ontology labels must be unique")
        if (self.dependency_lock_path is None) != (self.dependency_lock_sha256 is None):
            raise ValueError(
                "dependency_lock_path and dependency_lock_sha256 must be supplied together"
            )
        return self

    def resolve_artifact_path(self, root: Path) -> Path:
        try:
            root_resolved = Path(root).expanduser().resolve()
            lexical = root_resolved / Path(self.artifact_path)
            if any(part.is_symlink() for part in (lexical, *lexical.parents)):
                raise ArtifactVerificationError("artifact path contains a symlink component")
            candidate = lexical.resolve()
        except (OSError, TypeError) as exc:
            raise ArtifactVerificationError("unable to resolve artifact root/path") from exc
        try:
            candidate.relative_to(root_resolved)
        except ValueError as exc:
            raise ArtifactVerificationError("artifact path escapes the supplied root") from exc
        return candidate

    def resolve_dependency_lock_path(self, root: Path) -> Path | None:
        if self.dependency_lock_path is None:
            return None
        try:
            root_resolved = Path(root).expanduser().resolve()
            lexical = root_resolved / Path(self.dependency_lock_path)
            if any(part.is_symlink() for part in (lexical, *lexical.parents)):
                raise ArtifactVerificationError("dependency lock path contains a symlink component")
            candidate = lexical.resolve()
        except (OSError, TypeError) as exc:
            raise ArtifactVerificationError("unable to resolve dependency lock root/path") from exc
        try:
            candidate.relative_to(root_resolved)
        except ValueError as exc:
            raise ArtifactVerificationError(
                "dependency lock path escapes the supplied root"
            ) from exc
        return candidate


class ArtifactVerification(StrictModel):
    """Immutable receipt returned after local artifact verification."""

    manifest_sha256: str = Field(pattern=SHA256.pattern)
    artifact_id: str = Field(min_length=2, max_length=128)
    artifact_path: str = Field(min_length=1)
    artifact_sha256: str = Field(pattern=SHA256.pattern)
    artifact_size_bytes: int = Field(ge=1)
    dependency_lock_sha256: str | None = Field(default=None, pattern=SHA256.pattern)
    verified: bool = True

    @field_validator("verified", mode="before")
    @classmethod
    def verified_must_be_boolean(cls, value: object) -> bool:
        return validate_bool(value)

    @model_validator(mode="after")
    def must_be_verified(self) -> ArtifactVerification:
        if not self.verified:
            raise ValueError("an artifact verification receipt must be verified")
        return self


def _hash_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    total = 0
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
                total += len(chunk)
    except OSError as exc:
        raise ArtifactVerificationError(f"unable to read artifact: {path}") from exc
    return digest.hexdigest(), total


def validate_artifact_manifest(value: object) -> ModelArtifactManifest:
    """Validate an in-memory manifest using the strict public contract."""

    return ModelArtifactManifest.model_validate(value)


def load_artifact_manifest(path: Path) -> ModelArtifactManifest:
    """Load and strictly validate a local JSON manifest."""

    try:
        payload = load_strict_json(Path(path))
    except (OSError, TypeError, UnicodeError, ValueError) as exc:
        raise ArtifactVerificationError(f"unable to load artifact manifest: {path}") from exc
    return validate_artifact_manifest(payload)


def verify_artifact_manifest(
    manifest: ModelArtifactManifest | object,
    *,
    artifact_root: Path,
) -> ArtifactVerification:
    """Verify checkpoint and optional dependency-lock hashes below ``artifact_root``.

    The function never accepts URLs and never performs network I/O. A missing
    file, symlink escape, size mismatch, or hash mismatch raises an
    :class:`ArtifactVerificationError`.
    """

    validated = (
        manifest
        if isinstance(manifest, ModelArtifactManifest)
        else validate_artifact_manifest(manifest)
    )
    try:
        root = Path(artifact_root).expanduser().resolve()
    except (OSError, TypeError) as exc:
        raise ArtifactVerificationError("unable to resolve artifact root") from exc
    if not root.is_dir():
        raise ArtifactVerificationError(f"artifact root is not a directory: {artifact_root}")
    artifact_path = validated.resolve_artifact_path(root)
    if not artifact_path.is_file():
        raise ArtifactVerificationError(f"artifact file does not exist: {validated.artifact_path}")
    observed_hash, observed_size = _hash_file(artifact_path)
    if validated.artifact_size_bytes is not None and observed_size != validated.artifact_size_bytes:
        raise ArtifactVerificationError(
            f"artifact size mismatch: expected {validated.artifact_size_bytes}, got {observed_size}"
        )
    if observed_hash != validated.artifact_sha256:
        raise ArtifactVerificationError(
            f"artifact SHA-256 mismatch: expected {validated.artifact_sha256}, got {observed_hash}"
        )
    observed_lock_hash: str | None = None
    lock_path = validated.resolve_dependency_lock_path(root)
    if lock_path is not None:
        if not lock_path.is_file():
            raise ArtifactVerificationError(
                f"dependency lock file does not exist: {validated.dependency_lock_path}"
            )
        observed_lock_hash, _ = _hash_file(lock_path)
        expected_lock_hash = validated.dependency_lock_sha256
        if expected_lock_hash is None or observed_lock_hash != expected_lock_hash:
            raise ArtifactVerificationError(
                "dependency lock SHA-256 mismatch: "
                f"expected {expected_lock_hash}, got {observed_lock_hash}"
            )
    return ArtifactVerification(
        manifest_sha256=canonical_sha256(validated.model_dump(mode="json")),
        artifact_id=validated.artifact_id,
        artifact_path=validated.artifact_path,
        artifact_sha256=observed_hash,
        artifact_size_bytes=observed_size,
        dependency_lock_sha256=observed_lock_hash,
        verified=True,
    )


InputT_contra = TypeVar("InputT_contra", contravariant=True)
OutputT_co = TypeVar("OutputT_co", covariant=True)


@runtime_checkable
class ModelAdapter(Protocol, Generic[InputT_contra, OutputT_co]):
    """Minimal adapter seam; model frameworks remain optional dependencies."""

    @property
    def adapter_id(self) -> str: ...

    @property
    def manifest(self) -> ModelArtifactManifest: ...

    def infer(self, frame: InputT_contra) -> OutputT_co:
        """Run one explicit inference call on an already-decoded frame."""
        ...


PerceptionAdapter = ModelAdapter
ArtifactManifest = ModelArtifactManifest
ArtifactFormat = str


class AdapterRegistry:
    """Thread-safe registry of explicitly constructed model adapters."""

    def __init__(self) -> None:
        self._adapters: dict[str, ModelAdapter[Any, Any]] = {}
        self._manifests: dict[str, ModelArtifactManifest] = {}
        self._receipts: dict[str, ArtifactVerification | None] = {}
        self._lock = threading.RLock()

    def register(
        self,
        adapter: ModelAdapter[Any, Any],
        *,
        artifact_root: Path | None = None,
        replace: bool = False,
    ) -> ArtifactVerification | None:
        """Register an adapter, optionally verifying its local artifact."""

        adapter_id = getattr(adapter, "adapter_id", None)
        manifest_value = getattr(adapter, "manifest", None)
        infer = getattr(adapter, "infer", None)
        if not isinstance(adapter_id, str) or not _IDENTIFIER.fullmatch(adapter_id):
            raise ValueError("adapter_id must be a lowercase stable identifier")
        if not callable(infer):
            raise TypeError("adapter must expose a callable infer(frame) method")
        manifest = validate_artifact_manifest(manifest_value)
        if manifest.artifact_id != adapter_id:
            raise ValueError("adapter_id must match manifest.artifact_id")
        receipt = (
            verify_artifact_manifest(manifest, artifact_root=artifact_root)
            if artifact_root is not None
            else None
        )
        with self._lock:
            if adapter_id in self._adapters and not replace:
                raise KeyError(f"adapter already registered: {adapter_id}")
            self._adapters[adapter_id] = adapter
            self._manifests[adapter_id] = manifest
            self._receipts[adapter_id] = receipt
        return receipt

    def unregister(self, adapter_id: str) -> ModelAdapter[Any, Any]:
        with self._lock:
            try:
                adapter = self._adapters.pop(adapter_id)
            except KeyError as exc:
                raise KeyError(f"adapter is not registered: {adapter_id}") from exc
            self._manifests.pop(adapter_id, None)
            self._receipts.pop(adapter_id, None)
            return adapter

    def get(self, adapter_id: str) -> ModelAdapter[Any, Any]:
        with self._lock:
            try:
                return self._adapters[adapter_id]
            except KeyError as exc:
                raise KeyError(f"adapter is not registered: {adapter_id}") from exc

    def manifest(self, adapter_id: str) -> ModelArtifactManifest:
        with self._lock:
            try:
                return self._manifests[adapter_id]
            except KeyError as exc:
                raise KeyError(f"adapter is not registered: {adapter_id}") from exc

    def verification(self, adapter_id: str) -> ArtifactVerification | None:
        with self._lock:
            try:
                return self._receipts[adapter_id]
            except KeyError as exc:
                raise KeyError(f"adapter is not registered: {adapter_id}") from exc

    def is_verified(self, adapter_id: str) -> bool:
        return self.verification(adapter_id) is not None

    def require_verified(self, adapter_id: str) -> ModelAdapter[Any, Any]:
        with self._lock:
            try:
                adapter = self._adapters[adapter_id]
                receipt = self._receipts[adapter_id]
            except KeyError as exc:
                raise KeyError(f"adapter is not registered: {adapter_id}") from exc
            if receipt is None:
                raise ArtifactVerificationError(
                    f"adapter {adapter_id!r} has no verified artifact receipt"
                )
            return adapter

    def for_task(self, task: TaskKind | str) -> tuple[ModelAdapter[Any, Any], ...]:
        try:
            task_kind = task if isinstance(task, TaskKind) else TaskKind(task)
        except ValueError as exc:
            raise ValueError(f"unsupported task: {task}") from exc
        with self._lock:
            return tuple(
                self._adapters[adapter_id]
                for adapter_id in self._adapters
                if task_kind in self._manifests[adapter_id].tasks
            )

    def names(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._adapters)

    def __contains__(self, adapter_id: object) -> bool:
        with self._lock:
            return adapter_id in self._adapters

    def __iter__(self) -> Iterator[str]:
        return iter(self.names())

    def __len__(self) -> int:
        with self._lock:
            return len(self._adapters)


__all__ = [
    "MODEL_ARTIFACT_SCHEMA",
    "AdapterRegistry",
    "ArtifactFormat",
    "ArtifactInputSpec",
    "ArtifactManifest",
    "ArtifactOutputSpec",
    "ArtifactPreprocessingSpec",
    "ArtifactRuntimeSpec",
    "ArtifactVerification",
    "ArtifactVerificationError",
    "ModelAdapter",
    "ModelArtifactManifest",
    "OntologyClass",
    "PerceptionAdapter",
    "load_artifact_manifest",
    "validate_artifact_manifest",
    "verify_artifact_manifest",
]
