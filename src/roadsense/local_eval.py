"""Fail-closed evaluation of operator-provided local sequence data.

The Pages fixture is intentionally data-free.  This module is the explicit
local-data path for operator-run evaluation: it reads only files named by a
``roadsense.local-evaluation/v1`` JSON specification, validates complete
sequence boundaries, and delegates the actual metric calculations to the
pure evaluators in :mod:`roadsense.metrics`.

No URL is fetched and no dataset/model is downloaded.  Missing files,
ambiguous split membership, malformed records, and incompatible masks raise
``LocalEvaluationError`` before a report is produced.
"""

from __future__ import annotations

import hashlib
import math
import operator
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlparse

import numpy as np
from numpy.typing import NDArray

from roadsense.adapters import (
    ArtifactVerification,
    ArtifactVerificationError,
    ModelArtifactManifest,
    load_artifact_manifest,
    verify_artifact_manifest,
)
from roadsense.contracts import (
    DatasetManifest,
    Detection,
    EvaluationReport,
    EvidenceLevel,
    FrameRecord,
    TaskKind,
)
from roadsense.evidence import compute_report_id
from roadsense.json_io import canonical_sha256, load_strict_json
from roadsense.metrics import evaluate_detection, evaluate_segmentation, evaluate_tracking

LOCAL_EVALUATION_SCHEMA = "roadsense.local-evaluation/v1"
SEQUENCE_BUNDLE_SCHEMA = "roadsense.sequence-bundle/v1"
MAX_SEQUENCE_ID_LENGTH = 256
MAX_MASK_ELEMENTS = 64_000_000
MAX_MASK_FILE_BYTES = 512 * 1024 * 1024
MAX_JSON_FILE_BYTES = 256 * 1024 * 1024


class LocalEvaluationError(ValueError):
    """An invalid or incomplete local evaluation input.

    The exception intentionally contains an operator-facing remediation.  A
    CLI caller can print it without exposing a traceback or silently falling
    back to the synthetic fixture.
    """


@dataclass(frozen=True, slots=True)
class LocalEvaluationSpec:
    """Validated paths and protocol options from a local evaluation spec."""

    spec_path: Path
    payload: Mapping[str, Any]
    dataset_manifest_path: Path
    split: str
    split_sequences: Mapping[str, tuple[str, ...]]
    tasks: tuple[TaskKind, ...]
    ground_truth_path: Path
    predictions_path: Path
    artifact_manifest_path: Path | None
    artifact_root: Path | None
    segmentation_truth_paths: Mapping[str, Path]
    segmentation_prediction_paths: Mapping[str, Path]
    segmentation_num_classes: int | None
    segmentation_ignore_index: int
    detection_iou_threshold: float
    tracking_iou_threshold: float


@dataclass(frozen=True, slots=True)
class LocalEvaluationCase:
    """Loaded, aligned local records ready for metric evaluation."""

    spec: LocalEvaluationSpec
    dataset_manifest: DatasetManifest
    manifest_payload: Mapping[str, Any]
    truth_sequences: Mapping[str, tuple[FrameRecord, ...]]
    prediction_sequences: Mapping[str, tuple[FrameRecord, ...]]
    truth_masks: Mapping[str, NDArray[np.integer[Any]]]
    prediction_masks: Mapping[str, NDArray[np.integer[Any]]]
    input_hashes: Mapping[str, Mapping[str, object]]
    artifact_manifest: ModelArtifactManifest | None
    artifact_verification: ArtifactVerification | None


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise LocalEvaluationError(f"{name} must be a JSON object")
    return cast(Mapping[str, Any], value)


def _exact_keys(value: Mapping[str, Any], allowed: set[str], name: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise LocalEvaluationError(f"{name} contains unknown field(s): {', '.join(unknown)}")


def _non_empty_string(value: object, name: str, *, max_length: int = 2_048) -> str:
    if not isinstance(value, str) or not value.strip():
        raise LocalEvaluationError(f"{name} must be a non-empty string")
    if len(value) > max_length:
        raise LocalEvaluationError(f"{name} exceeds {max_length} characters")
    return value


def _integer(value: object, name: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool):
        raise LocalEvaluationError(f"{name} must be an integer")
    try:
        result = operator.index(cast(Any, value))
    except (TypeError, ValueError) as exc:
        raise LocalEvaluationError(f"{name} must be an integer") from exc
    if minimum is not None and result < minimum:
        raise LocalEvaluationError(f"{name} must be >= {minimum}")
    return int(result)


def _threshold(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LocalEvaluationError(f"{name} must be a finite number in [0, 1]")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise LocalEvaluationError(f"{name} must be a finite number in [0, 1]")
    return result


def _resolve_local_path(raw: object, *, base: Path, name: str) -> Path:
    """Resolve a local path and reject network-style URLs.

    Relative paths are resolved from the directory containing the evaluation
    spec.  Absolute paths are allowed so a licensed dataset may live outside
    the repository.  URI schemes are never fetched and are rejected with an
    actionable message.
    """

    text = _non_empty_string(raw, name)
    parsed = urlparse(text)
    if parsed.scheme and not (len(parsed.scheme) == 1 and len(text) >= 2 and text[1] == ":"):
        raise LocalEvaluationError(
            f"{name} must be a local filesystem path; URLs are never downloaded"
        )
    path = Path(text).expanduser()
    resolved = path.resolve() if path.is_absolute() else (base / path).resolve()
    if not resolved.exists():
        raise LocalEvaluationError(
            f"{name} does not exist: {resolved}. Place the licensed local data there; "
            "this command never downloads datasets."
        )
    if not resolved.is_file():
        raise LocalEvaluationError(f"{name} must point to a file: {resolved}")
    return resolved


def _resolve_local_directory(raw: object, *, base: Path, name: str) -> Path:
    """Resolve an operator-provided local directory without network access."""

    text = _non_empty_string(raw, name)
    parsed = urlparse(text)
    if parsed.scheme and not (len(parsed.scheme) == 1 and len(text) >= 2 and text[1] == ":"):
        raise LocalEvaluationError(
            f"{name} must be a local filesystem path; URLs are never downloaded"
        )
    path = Path(text).expanduser()
    resolved = path.resolve() if path.is_absolute() else (base / path).resolve()
    if not resolved.exists():
        raise LocalEvaluationError(
            f"{name} does not exist: {resolved}. Place the verified artifact root there; "
            "this command never downloads model files."
        )
    if not resolved.is_dir():
        raise LocalEvaluationError(f"{name} must point to a directory: {resolved}")
    return resolved


def _sequence_id(value: object, name: str) -> str:
    text = _non_empty_string(value, name, max_length=MAX_SEQUENCE_ID_LENGTH)
    if any(character in text for character in "\\/\r\n\x00"):
        raise LocalEvaluationError(f"{name} contains an invalid path/control character")
    return text


def _parse_tasks(value: object) -> tuple[TaskKind, ...]:
    if not isinstance(value, list) or not value:
        raise LocalEvaluationError("tasks must be a non-empty array")
    parsed: list[TaskKind] = []
    for index, item in enumerate(value):
        try:
            task = TaskKind(_non_empty_string(item, f"tasks[{index}]", max_length=32))
        except ValueError as exc:
            raise LocalEvaluationError(f"tasks[{index}] is not a supported task") from exc
        if task in parsed:
            raise LocalEvaluationError(f"tasks contains duplicate task: {task.value}")
        parsed.append(task)
    return tuple(parsed)


def _parse_split_sequences(value: object) -> Mapping[str, tuple[str, ...]]:
    mapping = _mapping(value, "split_sequences")
    if not mapping:
        raise LocalEvaluationError("split_sequences must contain at least one split")
    result: dict[str, tuple[str, ...]] = {}
    all_ids: set[str] = set()
    for raw_split, raw_ids in mapping.items():
        split = _non_empty_string(raw_split, "split name", max_length=128)
        if not isinstance(raw_ids, list) or not raw_ids:
            raise LocalEvaluationError(f"split_sequences[{split!r}] must be a non-empty array")
        ids = tuple(_sequence_id(item, f"split_sequences[{split!r}][]") for item in raw_ids)
        if len(set(ids)) != len(ids):
            raise LocalEvaluationError(f"split_sequences[{split!r}] contains duplicate IDs")
        overlap = all_ids.intersection(ids)
        if overlap:
            names = ", ".join(sorted(overlap))
            raise LocalEvaluationError(
                f"sequence IDs must be disjoint across splits; repeated: {names}"
            )
        all_ids.update(ids)
        result[split] = ids
    return result


def load_local_spec(path: Path | str) -> LocalEvaluationSpec:
    """Load and validate a local evaluation specification without network I/O."""

    spec_path = Path(path).expanduser().resolve()
    if not spec_path.exists():
        raise LocalEvaluationError(
            f"evaluation spec does not exist: {spec_path}; no local data was found"
        )
    if not spec_path.is_file():
        raise LocalEvaluationError(f"evaluation spec must be a file: {spec_path}")
    payload = _load_bounded_json(spec_path, role=f"evaluation spec {spec_path}")
    root = _mapping(payload, "evaluation spec")
    _exact_keys(
        root,
        {
            "schema_version",
            "dataset_manifest",
            "split",
            "split_sequences",
            "tasks",
            "ground_truth",
            "predictions",
            "model_artifact",
            "segmentation",
            "protocols",
        },
        "evaluation spec",
    )
    if root.get("schema_version") != LOCAL_EVALUATION_SCHEMA:
        raise LocalEvaluationError(f"schema_version must be {LOCAL_EVALUATION_SCHEMA!r}")
    split = _non_empty_string(root.get("split"), "split", max_length=128)
    split_sequences = _parse_split_sequences(root.get("split_sequences"))
    if split not in split_sequences:
        raise LocalEvaluationError(f"split {split!r} is missing from split_sequences")
    tasks = _parse_tasks(root.get("tasks"))
    base = spec_path.parent
    dataset_manifest_path = _resolve_local_path(
        root.get("dataset_manifest"), base=base, name="dataset_manifest"
    )
    ground_truth_path = _resolve_local_path(
        root.get("ground_truth"), base=base, name="ground_truth"
    )
    predictions_path = _resolve_local_path(root.get("predictions"), base=base, name="predictions")

    artifact_manifest_path: Path | None = None
    artifact_root: Path | None = None
    raw_artifact = root.get("model_artifact")
    if raw_artifact is not None:
        artifact = _mapping(raw_artifact, "model_artifact")
        _exact_keys(artifact, {"manifest", "root"}, "model_artifact")
        artifact_manifest_path = _resolve_local_path(
            artifact.get("manifest"), base=base, name="model_artifact.manifest"
        )
        artifact_root = _resolve_local_directory(
            artifact.get("root"), base=base, name="model_artifact.root"
        )

    protocols = _mapping(root.get("protocols", {}), "protocols")
    _exact_keys(protocols, {"detection_iou_threshold", "tracking_iou_threshold"}, "protocols")
    detection_iou_threshold = _threshold(
        protocols.get("detection_iou_threshold", 0.5), "protocols.detection_iou_threshold"
    )
    tracking_iou_threshold = _threshold(
        protocols.get("tracking_iou_threshold", 0.5), "protocols.tracking_iou_threshold"
    )

    segmentation_truth_paths: dict[str, Path] = {}
    segmentation_prediction_paths: dict[str, Path] = {}
    segmentation_num_classes: int | None = None
    segmentation_ignore_index = 255
    segmentation = _mapping(root.get("segmentation", {}), "segmentation")
    _exact_keys(
        segmentation,
        {"ground_truth", "predictions", "num_classes", "ignore_index"},
        "segmentation",
    )
    if TaskKind.SEGMENTATION in tasks:
        segmentation_num_classes = _integer(
            segmentation.get("num_classes"), "segmentation.num_classes", minimum=2
        )
        if segmentation_num_classes > 10_000:
            raise LocalEvaluationError("segmentation.num_classes must be <= 10000")
        if segmentation_num_classes * segmentation_num_classes > 4_000_000:
            raise LocalEvaluationError(
                "segmentation.num_classes is too large for the dense confusion matrix"
            )
        segmentation_ignore_index = _integer(
            segmentation.get("ignore_index", 255), "segmentation.ignore_index"
        )
        raw_truth_masks = _mapping(segmentation.get("ground_truth"), "segmentation.ground_truth")
        raw_prediction_masks = _mapping(segmentation.get("predictions"), "segmentation.predictions")
        selected_ids = set(split_sequences[split])
        if set(raw_truth_masks) != selected_ids or set(raw_prediction_masks) != selected_ids:
            raise LocalEvaluationError(
                "segmentation mask maps must contain exactly the selected sequence IDs"
            )
        for sequence_id in split_sequences[split]:
            segmentation_truth_paths[sequence_id] = _resolve_local_path(
                raw_truth_masks[sequence_id],
                base=base,
                name=f"segmentation.ground_truth[{sequence_id!r}]",
            )
            segmentation_prediction_paths[sequence_id] = _resolve_local_path(
                raw_prediction_masks[sequence_id],
                base=base,
                name=f"segmentation.predictions[{sequence_id!r}]",
            )
    elif segmentation:
        raise LocalEvaluationError("segmentation options are only valid when task is selected")

    return LocalEvaluationSpec(
        spec_path=spec_path,
        payload=root,
        dataset_manifest_path=dataset_manifest_path,
        split=split,
        split_sequences=split_sequences,
        tasks=tasks,
        ground_truth_path=ground_truth_path,
        predictions_path=predictions_path,
        artifact_manifest_path=artifact_manifest_path,
        artifact_root=artifact_root,
        segmentation_truth_paths=segmentation_truth_paths,
        segmentation_prediction_paths=segmentation_prediction_paths,
        segmentation_num_classes=segmentation_num_classes,
        segmentation_ignore_index=segmentation_ignore_index,
        detection_iou_threshold=detection_iou_threshold,
        tracking_iou_threshold=tracking_iou_threshold,
    )


def _load_manifest(path: Path) -> tuple[DatasetManifest, Mapping[str, Any]]:
    try:
        payload = _load_bounded_json(path, role=f"dataset manifest {path}")
        manifest = DatasetManifest.model_validate(payload)
    except (OSError, UnicodeError, ValueError, TypeError) as exc:
        raise LocalEvaluationError(f"dataset manifest validation failed: {exc}") from exc
    return manifest, _mapping(payload, "dataset manifest")


def _load_sequence_bundle(path: Path, *, role: str) -> Mapping[str, tuple[FrameRecord, ...]]:
    payload = _load_bounded_json(path, role=f"{role} bundle {path}")
    root = _mapping(payload, f"{role} bundle")
    _exact_keys(root, {"schema_version", "sequences"}, f"{role} bundle")
    if root.get("schema_version") != SEQUENCE_BUNDLE_SCHEMA:
        raise LocalEvaluationError(
            f"{role} bundle schema_version must be {SEQUENCE_BUNDLE_SCHEMA!r}"
        )
    raw_sequences = root.get("sequences")
    if not isinstance(raw_sequences, list) or not raw_sequences:
        raise LocalEvaluationError(f"{role} bundle must contain non-empty sequences")
    result: dict[str, tuple[FrameRecord, ...]] = {}
    for sequence_index, raw_sequence in enumerate(raw_sequences):
        sequence = _mapping(raw_sequence, f"{role} sequence[{sequence_index}]")
        _exact_keys(sequence, {"sequence_id", "frames"}, f"{role} sequence[{sequence_index}]")
        sequence_id = _sequence_id(
            sequence.get("sequence_id"), f"{role} sequence[{sequence_index}].sequence_id"
        )
        if sequence_id in result:
            raise LocalEvaluationError(f"{role} bundle repeats sequence ID {sequence_id!r}")
        raw_frames = sequence.get("frames")
        if not isinstance(raw_frames, list) or not raw_frames:
            raise LocalEvaluationError(f"{role} sequence {sequence_id!r} has no frames")
        frames: list[FrameRecord] = []
        for frame_index, raw_frame in enumerate(raw_frames):
            if not isinstance(raw_frame, Mapping):
                raise LocalEvaluationError(
                    f"{role} sequence {sequence_id!r} frame[{frame_index}] must be an object"
                )
            try:
                frame = FrameRecord.model_validate(raw_frame)
            except (TypeError, ValueError) as exc:
                raise LocalEvaluationError(
                    f"{role} sequence {sequence_id!r} frame[{frame_index}] is invalid: {exc}"
                ) from exc
            if frames:
                previous = frames[-1]
                if frame.frame_index <= previous.frame_index:
                    raise LocalEvaluationError(
                        f"{role} sequence {sequence_id!r} frame indices must be strictly increasing"
                    )
                if frame.timestamp_ms < previous.timestamp_ms:
                    raise LocalEvaluationError(
                        f"{role} sequence {sequence_id!r} timestamps must be monotonic"
                    )
            frames.append(frame)
        result[sequence_id] = tuple(frames)
    return result


def _load_mask(
    path: Path, *, role: str, sequence_id: str, frame_count: int
) -> NDArray[np.integer[Any]]:
    if path.suffix.lower() != ".npy":
        raise LocalEvaluationError(
            f"{role} mask for sequence {sequence_id!r} must be a .npy array "
            "(pickle/object arrays are not accepted)"
        )
    try:
        file_size = path.stat().st_size
    except OSError as exc:
        raise LocalEvaluationError(
            f"{role} mask for sequence {sequence_id!r} cannot be inspected: {exc}"
        ) from exc
    if file_size > MAX_MASK_FILE_BYTES:
        raise LocalEvaluationError(
            f"{role} mask for sequence {sequence_id!r} exceeds the "
            f"{MAX_MASK_FILE_BYTES} byte safety limit"
        )
    try:
        # Memory-map first so the shape/element guard runs before copying a
        # large local array into RAM.  Only plain .npy files are accepted.
        array = np.load(path, allow_pickle=False, mmap_mode="r")
    except (OSError, ValueError, MemoryError) as exc:
        raise LocalEvaluationError(
            f"{role} mask for sequence {sequence_id!r} cannot be read: {exc}"
        ) from exc
    if not isinstance(array, np.ndarray):
        raise LocalEvaluationError(f"{role} mask for sequence {sequence_id!r} is not a .npy array")
    if array.ndim != 3 or array.shape[0] != frame_count or array.shape[1] < 1 or array.shape[2] < 1:
        raise LocalEvaluationError(
            f"{role} mask for sequence {sequence_id!r} must have shape "
            f"(frames,height,width) with frames={frame_count}; got {array.shape}"
        )
    if array.size > MAX_MASK_ELEMENTS:
        raise LocalEvaluationError(
            f"{role} mask for sequence {sequence_id!r} exceeds the "
            f"{MAX_MASK_ELEMENTS} element safety limit"
        )
    if not np.issubdtype(array.dtype, np.integer):
        raise LocalEvaluationError(
            f"{role} mask for sequence {sequence_id!r} must use an integer dtype"
        )
    return cast(NDArray[np.integer[Any]], array)


def _sha256_file(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return {"sha256": digest.hexdigest(), "bytes": size}


def _load_bounded_json(path: Path, *, role: str) -> Any:
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise LocalEvaluationError(f"{role} cannot be inspected: {exc}") from exc
    if size > MAX_JSON_FILE_BYTES:
        raise LocalEvaluationError(f"{role} exceeds the {MAX_JSON_FILE_BYTES} byte safety limit")
    try:
        return load_strict_json(path)
    except (OSError, UnicodeError, ValueError) as exc:
        raise LocalEvaluationError(f"{role} cannot be read: {exc}") from exc


def load_local_evaluation(path: Path | str) -> LocalEvaluationCase:
    """Load all local inputs referenced by ``path`` and validate split alignment."""

    spec = load_local_spec(path)
    manifest, manifest_payload = _load_manifest(spec.dataset_manifest_path)
    if spec.split not in manifest.splits:
        raise LocalEvaluationError(
            f"split {spec.split!r} is not declared by dataset manifest {manifest.dataset_name!r}"
        )
    manifest_tasks = set(manifest.tasks)
    if any(task not in manifest_tasks for task in spec.tasks):
        missing = ", ".join(task.value for task in spec.tasks if task not in manifest_tasks)
        raise LocalEvaluationError(f"evaluation tasks are absent from dataset manifest: {missing}")
    artifact_manifest: ModelArtifactManifest | None = None
    artifact_verification: ArtifactVerification | None = None
    if spec.artifact_manifest_path is not None or spec.artifact_root is not None:
        if spec.artifact_manifest_path is None or spec.artifact_root is None:
            # This is defensive (the spec parser currently requires both), but
            # keeps the case contract fail-closed if it is constructed directly.
            raise LocalEvaluationError(
                "model_artifact requires both a manifest path and an artifact root"
            )
        try:
            artifact_manifest = load_artifact_manifest(spec.artifact_manifest_path)
            artifact_verification = verify_artifact_manifest(
                artifact_manifest, artifact_root=spec.artifact_root
            )
        except (ArtifactVerificationError, OSError, TypeError, ValueError) as exc:
            raise LocalEvaluationError(f"model artifact verification failed: {exc}") from exc
        missing_artifact_tasks = [
            task.value for task in spec.tasks if task not in set(artifact_manifest.tasks)
        ]
        if missing_artifact_tasks:
            raise LocalEvaluationError(
                "verified model artifact does not declare task(s): "
                + ", ".join(missing_artifact_tasks)
            )
    elif manifest.frozen or manifest.evaluation_authorized:
        raise LocalEvaluationError(
            "authorized or frozen evaluation requires a verified model_artifact block"
        )
    selected_ids = tuple(spec.split_sequences[spec.split])
    truth = _load_sequence_bundle(spec.ground_truth_path, role="ground_truth")
    predictions = _load_sequence_bundle(spec.predictions_path, role="predictions")
    expected = set(selected_ids)
    if set(truth) != expected or set(predictions) != expected:
        raise LocalEvaluationError(
            "ground_truth and predictions bundles must contain exactly the selected split sequences"
        )
    for sequence_id in selected_ids:
        truth_frames = truth[sequence_id]
        prediction_frames = predictions[sequence_id]
        if len(truth_frames) != len(prediction_frames):
            raise LocalEvaluationError(f"sequence {sequence_id!r} has mismatched frame counts")
        for index, (truth_frame, prediction_frame) in enumerate(
            zip(truth_frames, prediction_frames, strict=True)
        ):
            if (
                truth_frame.frame_index != prediction_frame.frame_index
                or truth_frame.timestamp_ms != prediction_frame.timestamp_ms
                or truth_frame.image_size != prediction_frame.image_size
            ):
                raise LocalEvaluationError(
                    f"sequence {sequence_id!r} frame[{index}] is not aligned between "
                    "ground_truth and predictions"
                )
    if TaskKind.DETECTION in spec.tasks:
        empty_sequences = [
            sequence_id
            for sequence_id in selected_ids
            if not any(truth_frame.detections for truth_frame in truth[sequence_id])
        ]
        if empty_sequences:
            names = ", ".join(empty_sequences)
            raise LocalEvaluationError(
                "detection ground truth contains no objects in sequence(s) "
                f"{names}; refusing to report an undefined per-sequence AP"
            )

    if artifact_manifest is not None:
        ontology_ids = {item.category_id for item in artifact_manifest.ontology}
        for role, sequences in (("ground_truth", truth), ("predictions", predictions)):
            unknown_ids = sorted(
                {
                    detection.category_id
                    for frames in sequences.values()
                    for frame in frames
                    for detection in frame.detections
                    if detection.category_id not in ontology_ids
                }
            )
            if unknown_ids:
                raise LocalEvaluationError(
                    f"{role} contains category IDs absent from the verified model ontology: "
                    + ", ".join(str(item) for item in unknown_ids)
                )

    truth_masks: dict[str, NDArray[np.integer[Any]]] = {}
    prediction_masks: dict[str, NDArray[np.integer[Any]]] = {}
    if TaskKind.SEGMENTATION in spec.tasks:
        for sequence_id in selected_ids:
            truth_masks[sequence_id] = _load_mask(
                spec.segmentation_truth_paths[sequence_id],
                role="ground_truth",
                sequence_id=sequence_id,
                frame_count=len(truth[sequence_id]),
            )
            prediction_masks[sequence_id] = _load_mask(
                spec.segmentation_prediction_paths[sequence_id],
                role="predictions",
                sequence_id=sequence_id,
                frame_count=len(predictions[sequence_id]),
            )
            if truth_masks[sequence_id].shape[1:] != prediction_masks[sequence_id].shape[1:]:
                raise LocalEvaluationError(
                    f"sequence {sequence_id!r} segmentation mask shapes do not match"
                )

    input_hashes: dict[str, Mapping[str, object]] = {
        "spec": _sha256_file(spec.spec_path),
        "dataset_manifest": _sha256_file(spec.dataset_manifest_path),
        "ground_truth": _sha256_file(spec.ground_truth_path),
        "predictions": _sha256_file(spec.predictions_path),
    }
    if spec.artifact_manifest_path is not None:
        input_hashes["model_artifact_manifest"] = _sha256_file(spec.artifact_manifest_path)
    if artifact_verification is not None:
        input_hashes["model_artifact_receipt"] = artifact_verification.model_dump(mode="json")
    if TaskKind.SEGMENTATION in spec.tasks:
        input_hashes["segmentation_ground_truth"] = {
            sequence_id: _sha256_file(spec.segmentation_truth_paths[sequence_id])
            for sequence_id in selected_ids
        }
        input_hashes["segmentation_predictions"] = {
            sequence_id: _sha256_file(spec.segmentation_prediction_paths[sequence_id])
            for sequence_id in selected_ids
        }
    return LocalEvaluationCase(
        spec=spec,
        dataset_manifest=manifest,
        manifest_payload=manifest_payload,
        truth_sequences=truth,
        prediction_sequences=predictions,
        truth_masks=truth_masks,
        prediction_masks=prediction_masks,
        input_hashes=input_hashes,
        artifact_manifest=artifact_manifest,
        artifact_verification=artifact_verification,
    )


def _renumber_frames(
    frames: Sequence[FrameRecord],
    *,
    frame_offset: int,
    timestamp_offset: int,
    track_id_map: Mapping[int, int] | None,
) -> tuple[FrameRecord, ...]:
    result: list[FrameRecord] = []
    for frame in frames:
        detections: list[Detection] = []
        for detection in frame.detections:
            if track_id_map is not None and detection.track_id is not None:
                detections.append(
                    detection.model_copy(update={"track_id": track_id_map[detection.track_id]})
                )
            else:
                detections.append(detection)
        result.append(
            frame.model_copy(
                update={
                    "frame_index": frame.frame_index + frame_offset,
                    "timestamp_ms": frame.timestamp_ms + timestamp_offset,
                    "detections": tuple(detections),
                }
            )
        )
    return tuple(result)


def _merge_sequences(
    sequences: Mapping[str, tuple[FrameRecord, ...]],
    ordered_ids: Sequence[str],
    *,
    track_id_maps: Mapping[str, Mapping[int, int]] | None = None,
) -> tuple[FrameRecord, ...]:
    """Concatenate complete sequences with deterministic index/ID namespaces."""

    result: list[FrameRecord] = []
    next_frame_offset = 0
    next_timestamp_offset = 0
    for sequence_id in ordered_ids:
        frames = sequences[sequence_id]
        max_frame_index = max(frame.frame_index for frame in frames)
        max_timestamp = max(frame.timestamp_ms for frame in frames)
        track_map = track_id_maps.get(sequence_id) if track_id_maps is not None else None
        if track_map is not None:
            expected_ids = {
                detection.track_id
                for frame in frames
                for detection in frame.detections
                if detection.track_id is not None
            }
            if not expected_ids.issubset(track_map):
                raise LocalEvaluationError(
                    f"track namespace is missing an ID in sequence {sequence_id!r}"
                )
        result.extend(
            _renumber_frames(
                frames,
                frame_offset=next_frame_offset,
                timestamp_offset=next_timestamp_offset,
                track_id_map=track_map,
            )
        )
        next_frame_offset += max_frame_index + 1
        # A one-millisecond gap prevents a timestamp regression when two local
        # sequences both start at zero, while preserving all within-sequence deltas.
        next_timestamp_offset += max_timestamp + 1
    return tuple(result)


def _build_track_namespace(
    truth: Mapping[str, tuple[FrameRecord, ...]],
    predictions: Mapping[str, tuple[FrameRecord, ...]],
    ordered_ids: Sequence[str],
) -> Mapping[str, Mapping[int, int]]:
    """Create one shared sequence namespace for truth and prediction IDs.

    The same original ID in one sequence maps to the same integer in both
    streams; IDs from different sequences are kept disjoint.  This avoids the
    common but serious mistake of independently renumbering truth and
    prediction IDs and accidentally turning an identity mismatch into a hit.
    """

    next_track_id = 1
    result: dict[str, Mapping[int, int]] = {}
    for sequence_id in ordered_ids:
        ids = sorted(
            {
                detection.track_id
                for frames in (truth[sequence_id], predictions[sequence_id])
                for frame in frames
                for detection in frame.detections
                if detection.track_id is not None
            }
        )
        if next_track_id + len(ids) - 1 > 1_000_000:
            raise LocalEvaluationError(
                "the selected split contains too many unique track IDs for the public contract"
            )
        result[sequence_id] = {
            track_id: next_track_id + index for index, track_id in enumerate(ids)
        }
        next_track_id += len(ids)
    return result


def _flatten_metrics(reports: Mapping[str, Mapping[str, object]]) -> dict[str, float]:
    flattened: dict[str, float] = {}
    if "detection" in reports:
        report = reports["detection"]
        flattened["detection_ap"] = float(cast(float, report["ap"]))
        flattened["detection_precision"] = float(cast(float, report["precision"]))
        flattened["detection_recall"] = float(cast(float, report["recall"]))
    if "segmentation" in reports:
        report = reports["segmentation"]
        flattened["segmentation_mean_iou"] = float(cast(float, report["mean_iou"]))
        flattened["segmentation_pixel_accuracy"] = float(cast(float, report["pixel_accuracy"]))
    if "tracking" in reports:
        report = reports["tracking"]
        flattened["tracking_mota"] = float(cast(float, report["mota"]))
        flattened["tracking_identity_f1"] = float(cast(float, report["identity_f1"]))
    if not flattened or any(not math.isfinite(value) for value in flattened.values()):
        raise LocalEvaluationError("evaluation produced no finite aggregate metrics")
    return flattened


def evaluate_local(path: Path | str) -> dict[str, object]:
    """Evaluate a validated local spec and return a hash-bound report.

    The returned report uses ``development`` evidence unless the dataset
    manifest explicitly marks a run as authorized and frozen.  It never
    upgrades fixture or missing-data inputs to benchmark evidence.
    """

    case = load_local_evaluation(path)
    selected_ids = tuple(case.spec.split_sequences[case.spec.split])
    track_namespace = (
        _build_track_namespace(case.truth_sequences, case.prediction_sequences, selected_ids)
        if TaskKind.TRACKING in case.spec.tasks
        else None
    )
    truth_merged = _merge_sequences(
        case.truth_sequences, selected_ids, track_id_maps=track_namespace
    )
    prediction_merged = _merge_sequences(
        case.prediction_sequences, selected_ids, track_id_maps=track_namespace
    )

    sequence_reports: dict[str, dict[str, Mapping[str, object]]] = {}
    aggregate_reports: dict[str, Mapping[str, object]] = {}
    for sequence_id in selected_ids:
        truth_frames = case.truth_sequences[sequence_id]
        prediction_frames = case.prediction_sequences[sequence_id]
        reports: dict[str, Mapping[str, object]] = {}
        try:
            if TaskKind.DETECTION in case.spec.tasks:
                reports["detection"] = evaluate_detection(
                    truth_frames,
                    prediction_frames,
                    iou_threshold=case.spec.detection_iou_threshold,
                )
            if TaskKind.TRACKING in case.spec.tasks:
                reports["tracking"] = evaluate_tracking(
                    truth_frames,
                    prediction_frames,
                    iou_threshold=case.spec.tracking_iou_threshold,
                )
            if TaskKind.SEGMENTATION in case.spec.tasks:
                reports["segmentation"] = evaluate_segmentation(
                    case.truth_masks[sequence_id],
                    case.prediction_masks[sequence_id],
                    num_classes=cast(int, case.spec.segmentation_num_classes),
                    ignore_index=case.spec.segmentation_ignore_index,
                )
        except (TypeError, ValueError, MemoryError) as exc:
            raise LocalEvaluationError(
                f"metric evaluation failed for sequence {sequence_id!r}: {exc}"
            ) from exc
        sequence_reports[sequence_id] = reports

    try:
        if TaskKind.DETECTION in case.spec.tasks:
            aggregate_reports["detection"] = evaluate_detection(
                truth_merged,
                prediction_merged,
                iou_threshold=case.spec.detection_iou_threshold,
            )
        if TaskKind.TRACKING in case.spec.tasks:
            aggregate_reports["tracking"] = evaluate_tracking(
                truth_merged,
                prediction_merged,
                iou_threshold=case.spec.tracking_iou_threshold,
            )
        if TaskKind.SEGMENTATION in case.spec.tasks:
            truth_masks = _concat_masks(case.truth_masks, selected_ids)
            prediction_masks = _concat_masks(case.prediction_masks, selected_ids)
            aggregate_reports["segmentation"] = evaluate_segmentation(
                truth_masks,
                prediction_masks,
                num_classes=cast(int, case.spec.segmentation_num_classes),
                ignore_index=case.spec.segmentation_ignore_index,
            )
    except (TypeError, ValueError, MemoryError) as exc:
        raise LocalEvaluationError(f"aggregate metric evaluation failed: {exc}") from exc

    metrics = _flatten_metrics(aggregate_reports)
    manifest_sha = canonical_sha256(dict(case.manifest_payload))
    evidence_level = (
        EvidenceLevel.FROZEN_EVALUATION
        if case.dataset_manifest.frozen
        else EvidenceLevel.DEVELOPMENT
    )
    details: dict[str, object] = {
        "dataset_name": case.dataset_manifest.dataset_name,
        "dataset_content_sha256": case.dataset_manifest.content_sha256,
        # The compact in-repository protocols are diagnostics.  Even an
        # authorized local run must not be presented as an official public
        # benchmark until a pinned independent evaluator is attached.
        "benchmark_claim_available": False,
        "split": case.spec.split,
        "sequence_ids": list(selected_ids),
        "tasks": [task.value for task in case.spec.tasks],
        "protocols": {task: dict(report) for task, report in aggregate_reports.items()},
        "sequence_reports": {
            sequence_id: {task: dict(report) for task, report in reports.items()}
            for sequence_id, reports in sequence_reports.items()
        },
        "input_hashes": dict(case.input_hashes),
        "spec_sha256": case.input_hashes["spec"]["sha256"],
        "model_artifact": (
            {
                "bound": True,
                "manifest_sha256": case.artifact_verification.manifest_sha256,
                "artifact_id": case.artifact_verification.artifact_id,
                "artifact_path": case.artifact_verification.artifact_path,
                "artifact_sha256": case.artifact_verification.artifact_sha256,
                "artifact_size_bytes": case.artifact_verification.artifact_size_bytes,
                "verified": case.artifact_verification.verified,
            }
            if case.artifact_verification is not None
            else {
                "bound": False,
                "verified": False,
                "claim_boundary": (
                    "No model-artifact manifest was supplied; this report binds the local "
                    "prediction files only and is not model provenance."
                ),
            }
        ),
    }
    report = EvaluationReport(
        schema_version="roadsense.evaluation-report/v1",
        protocol_id=LOCAL_EVALUATION_SCHEMA,
        evidence_level=evidence_level,
        dataset_manifest_sha256=manifest_sha,
        evaluation_authorized=case.dataset_manifest.evaluation_authorized,
        frozen=case.dataset_manifest.frozen,
        metrics=metrics,
        claim_boundary=(
            "Local operator-provided data evaluated with RoadSense compact protocols. "
            "These values are not official COCO, BDD100K, MOT, HOTA, or TrackEval metrics "
            "unless an independently pinned evaluator and authorization are recorded. "
            "A verified model-artifact receipt is included when a model provenance claim "
            "is intended; otherwise only prediction-file provenance is established."
        ),
    ).model_dump(mode="json")
    report["details"] = details
    report["report_id"] = compute_report_id(report)
    return report


def _concat_masks(
    masks: Mapping[str, NDArray[np.integer[Any]]], ordered_ids: Sequence[str]
) -> NDArray[np.integer[Any]]:
    if not ordered_ids:
        raise LocalEvaluationError("at least one sequence is required for segmentation")
    shapes = {masks[sequence_id].shape[1:] for sequence_id in ordered_ids}
    if len(shapes) != 1:
        raise LocalEvaluationError("all segmentation masks in a split must share height/width")
    return cast(
        NDArray[np.integer[Any]], np.concatenate([masks[key] for key in ordered_ids], axis=0)
    )


def fixture_dry_run_summary() -> dict[str, object]:
    """Return an explicit, non-benchmark fixture dry-run summary."""

    from roadsense.evidence import build_fixture_report

    report = build_fixture_report()
    return {
        "status": "ok",
        "source": "deterministic_geometric_fixture",
        "fixture_id": "roadsense-city-loop-v1",
        "evidence_level": report["evidence_level"],
        "evaluation_authorized": report["evaluation_authorized"],
        "frozen": report["frozen"],
        "benchmark_claim_available": False,
        "report_id": report["report_id"],
        "metrics": report["metrics"],
    }
