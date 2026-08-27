"""Finalize a reproducible BDD100K Detection benchmark receipt.

This stage is intentionally *aggregation only*.  It does not run ONNX
inference, invoke the BDD evaluator, download data, or include BDD media,
labels, predictions, model weights, or local paths in its output.  The caller
supplies the immutable receipts produced by the preparation, inference, and
two independent evaluation stages.  The final sanitized JSON is accepted as
a benchmark claim only after every provenance link and both evaluator runs
pass the fail-closed checks.

Typical invocation (all paths are local operator inputs)::

    python scripts/finalize_bdd100k_detection_benchmark.py \
      --source-receipt data/.../source-receipt.json \
      --dataset-manifest data/.../dataset-manifest.json \
      --split-inventory data/.../split-inventory.json \
      --image-manifest runs/frozen-image-manifest.json \
      --model-manifest runs/model-manifest.json \
      --inference-receipt runs/inference-receipt.json \
      --evaluation-a runs/eval-a/evaluation-receipt.json \
      --evaluation-b runs/eval-b/evaluation-receipt.json \
      --output benchmark-receipt.json

Only hashes and aggregate finite metrics cross the publication boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any, cast
from urllib.parse import urlsplit

from roadsense.bdd_benchmark import (
    BDD100K_DETECTION_BENCHMARK_SCHEMA,
    BDD100K_DETECTION_SCOPE,
    BDD100K_DEVKIT_COMMIT,
    BDD100K_DEVKIT_ID,
    BDD100K_DEVKIT_REPOSITORY,
    BDD100K_OFFICIAL_IMAGE_COUNT,
    BDD100K_OFFICIAL_IMAGES_MD5,
    BDD100K_OFFICIAL_LABELS_MD5,
    BDD100K_REQUIRED_EVALUATOR_PACKAGES,
    BDD100KBenchmarkReceiptError,
    build_bdd100k_detection_receipt,
)
from roadsense.json_io import canonical_sha256, load_strict_json, write_json_atomic

try:
    # Editable installs and pytest expose the repository root as a package
    # parent.  Direct ``python scripts/finalize_...py`` execution instead
    # places only ``scripts/`` on ``sys.path``; mirror the runner's sibling
    # fallback so the documented command works in both modes.
    from scripts.run_bdd100k_detection_benchmark import (
        FrozenManifestError,
        validate_frozen_image_manifest,
    )
except ModuleNotFoundError:  # pragma: no cover - exercised by direct script CLI
    from run_bdd100k_detection_benchmark import (  # type: ignore[no-redef]
        FrozenManifestError,
        validate_frozen_image_manifest,
    )

FINALIZE_SCHEMA = "roadsense.bdd100k-detection-finalize/v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9._-]{1,127}$")
_MAX_EVALUATOR_RESULT_BYTES = 16 * 1024 * 1024
_OFFICIAL_DETECTION_METRIC_KEYS = (
    "AP",
    "AP50",
    "AP75",
    "APs",
    "APm",
    "APl",
    "AR1",
    "AR10",
    "AR100",
    "ARs",
    "ARm",
    "ARl",
)

# Formal BDD100K Detection 2020 ``val`` provenance.  These values mirror the
# checks in ``prepare_bdd100k_detection``.  They are deliberately repeated
# here instead of trusting a caller-provided source receipt: finalization is a
# separate publication boundary and must reject a Kaggle/community label
# bundle even when it has the expected filename/layout.  Reduced synthetic
# fixtures used by unit tests are allowed below the official 10,000-image
# cardinality, but they cannot be promoted as the real lane.
BDD100K_OFFICIAL_SOURCE_PAGE = "https://bdd-data.berkeley.edu/"
BDD100K_OFFICIAL_ARCHIVE_ROLES = {
    "images_val_zip": "zip",
    "det_20_labels": "zip",
}


class BDD100KFinalizeError(ValueError):
    """Raised when benchmark evidence cannot be finalized safely."""


def _is_official_berkeley_url(value: object) -> bool:
    """Accept only HTTPS URLs on the exact credential-free portal origin."""

    if not isinstance(value, str) or value != value.strip():
        return False
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError:
        return False
    return (
        parsed.scheme == "https"
        and parsed.hostname == "bdd-data.berkeley.edu"
        and parsed.username is None
        and parsed.password is None
        and port in (None, 443)
    )


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
                size += len(chunk)
    except OSError as exc:
        raise BDD100KFinalizeError(f"cannot read evidence file: {path}") from exc
    return digest.hexdigest(), size


def _sha256_text(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha(value: object, field: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value) or value == "0" * 64:
        raise BDD100KFinalizeError(f"{field} must be a non-zero lowercase SHA-256 digest")
    return value


def _require_identifier(value: object, field: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise BDD100KFinalizeError(f"{field} must be a safe lowercase identifier")
    return value


def _finite_metrics(value: object, field: str) -> dict[str, float]:
    if not isinstance(value, Mapping) or not value:
        raise BDD100KFinalizeError(f"{field} must be a non-empty metrics object")
    metrics: dict[str, float] = {}
    for key, metric in value.items():
        if not isinstance(key, str) or not key or any(ord(char) < 32 for char in key):
            raise BDD100KFinalizeError(f"{field} contains an unsafe metric name")
        if isinstance(metric, bool) or not isinstance(metric, (int, float)):
            raise BDD100KFinalizeError(f"{field}.{key} must be numeric")
        number = float(metric)
        if not math.isfinite(number):
            raise BDD100KFinalizeError(f"{field}.{key} must be finite")
        metrics[key] = number
    return dict(sorted(metrics.items()))


def _load_source(
    source: Path | str | Mapping[str, Any], *, role: str
) -> tuple[dict[str, Any], Path | None]:
    """Load strict JSON and retain the parent directory only for local hash checks."""

    if isinstance(source, Mapping):
        return dict(source), None
    path = Path(source).expanduser().resolve()
    try:
        payload = load_strict_json(path)
    except (OSError, UnicodeError, TypeError, ValueError) as exc:
        raise BDD100KFinalizeError(f"unable to load {role}: {path.name}") from exc
    if not isinstance(payload, Mapping):
        raise BDD100KFinalizeError(f"{role} must contain a JSON object")
    return dict(payload), path.parent


def _check_optional_file_hash(
    *,
    base: Path | None,
    relative_name: object,
    expected_sha: object,
    field: str,
    required: bool = True,
) -> Path | None:
    """Verify a receipt's relative artifact when a path-backed receipt exists."""

    if not isinstance(relative_name, str) or not relative_name:
        raise BDD100KFinalizeError(f"{field} path must be relative")
    # Receipt paths are portable relative POSIX names.  Reject backslashes,
    # drive-qualified names, URI-like schemes, control characters, and dot
    # segments before asking the host OS to resolve them.  Without this
    # lexical check, a Windows path could be interpreted as an ordinary file
    # name when the finalizer runs on Linux (or vice versa).
    if any(ord(character) < 32 or ord(character) == 127 for character in relative_name):
        raise BDD100KFinalizeError(f"{field} path must be relative")
    if relative_name.replace("\\", "/") != relative_name:
        raise BDD100KFinalizeError(f"{field} path must use portable '/' separators")
    if re.match(r"^[A-Za-z]:", relative_name) or relative_name.startswith("/"):
        raise BDD100KFinalizeError(f"{field} path must be relative")
    parts = PurePosixPath(relative_name).parts
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise BDD100KFinalizeError(f"{field} path must not contain dot segments")
    if base is None:
        # In-memory receipts have no artifact directory to hash, but their
        # path fields still need to obey the same portable contract.
        return None
    path = (base / Path(*parts)).resolve()
    try:
        path.relative_to(base.resolve())
    except ValueError as exc:
        raise BDD100KFinalizeError(f"{field} path escapes its receipt directory") from exc
    if not path.is_file() or path.is_symlink():
        if required:
            raise BDD100KFinalizeError(f"{field} artifact is missing")
        return None
    observed, _size = _sha256_file(path)
    if observed != _require_sha(expected_sha, f"{field}.sha256"):
        raise BDD100KFinalizeError(f"{field} artifact hash does not match its receipt")
    return path


def _load_bound_official_detection_metrics(path: Path, *, expected_sha256: str) -> dict[str, float]:
    """Parse official aggregate metrics from the exact hash-bound result bytes.

    The evaluator receipt is not trusted to restate metrics faithfully.  Read
    the result artifact again, verify the digest over the same bytes that are
    parsed, reject duplicate keys, and require the complete Detection result
    schema emitted by the pinned Scalabel evaluator.
    """

    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise BDD100KFinalizeError("cannot read evaluator result artifact") from exc
    if not raw or len(raw) > _MAX_EVALUATOR_RESULT_BYTES:
        raise BDD100KFinalizeError("evaluator result artifact has an unsafe size")
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise BDD100KFinalizeError("evaluator result artifact hash does not match its receipt")

    def parse_constant(token: str) -> float:
        if token == "NaN":
            return float("nan")
        raise ValueError(f"unsupported non-finite JSON constant: {token}")

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = item
        return result

    try:
        payload = json.loads(
            raw.decode("utf-8", errors="strict"),
            parse_constant=parse_constant,
            object_pairs_hook=reject_duplicate_keys,
        )
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise BDD100KFinalizeError("evaluator result artifact is not valid official JSON") from exc
    if not isinstance(payload, Mapping):
        raise BDD100KFinalizeError("official evaluator result must be a JSON object")

    expected_keys = set(_OFFICIAL_DETECTION_METRIC_KEYS)
    observed_keys = set(payload)
    if observed_keys != expected_keys:
        missing = ", ".join(sorted(expected_keys - observed_keys)) or "none"
        unexpected = ", ".join(sorted(observed_keys - expected_keys)) or "none"
        raise BDD100KFinalizeError(
            "official evaluator result metric fields do not match the pinned Detection schema "
            f"(missing={missing}; unexpected={unexpected})"
        )

    metrics: dict[str, float] = {}
    for key in _OFFICIAL_DETECTION_METRIC_KEYS:
        rows = payload[key]
        if not isinstance(rows, list) or not rows:
            raise BDD100KFinalizeError(
                f"official evaluator result metric {key} must contain score rows"
            )
        overall_values = [
            row["OVERALL"] for row in rows if isinstance(row, Mapping) and "OVERALL" in row
        ]
        if len(overall_values) != 1:
            raise BDD100KFinalizeError(
                f"official evaluator result metric {key} must contain exactly one OVERALL value"
            )
        candidate = overall_values[0]
        if isinstance(candidate, bool) or not isinstance(candidate, (int, float)):
            raise BDD100KFinalizeError(
                f"official evaluator result metric {key}.OVERALL must be numeric"
            )
        number = float(candidate)
        if not math.isfinite(number):
            raise BDD100KFinalizeError(
                f"official evaluator result metric {key}.OVERALL must be finite"
            )
        metrics[key] = number
    return dict(sorted(metrics.items()))


def _archive_material(source_receipt: Mapping[str, Any]) -> list[dict[str, Any]]:
    archives = source_receipt.get("source_archives")
    if not isinstance(archives, list) or not archives:
        raise BDD100KFinalizeError("source receipt must contain source_archives")
    material: list[dict[str, Any]] = []
    for index, archive in enumerate(archives):
        if not isinstance(archive, Mapping):
            raise BDD100KFinalizeError(f"source_archives[{index}] must be an object")
        role = archive.get("role")
        digest = _require_sha(archive.get("sha256"), f"source_archives[{index}].sha256")
        size = archive.get("bytes")
        if isinstance(size, bool) or not isinstance(size, int) or size < 1:
            raise BDD100KFinalizeError(f"source_archives[{index}].bytes must be positive")
        package_format = archive.get("format")
        if not isinstance(role, str) or not role or not isinstance(package_format, str):
            raise BDD100KFinalizeError(f"source_archives[{index}] has invalid identity fields")
        record: dict[str, Any] = {
            "role": role,
            "format": package_format,
            "sha256": digest,
            "bytes": size,
        }
        # An official package MD5 is useful provenance, but it is not trusted
        # as the primary identity and is omitted when absent.
        if "official_package_md5" in archive:
            md5 = archive["official_package_md5"]
            if not isinstance(md5, str) or not re.fullmatch(r"[0-9a-fA-F]{32}", md5):
                raise BDD100KFinalizeError(
                    f"source_archives[{index}].official_package_md5 is invalid"
                )
            record["official_package_md5"] = md5.lower()
        material.append(record)
    return sorted(material, key=lambda item: (str(item["role"]), str(item["sha256"])))


def compute_archive_sha256(source_receipt: Mapping[str, Any]) -> str:
    """Compute a stable digest binding every operator-provided source archive."""

    return canonical_sha256(_archive_material(source_receipt))


def _validate_official_source_attestation(
    source_receipt: Mapping[str, Any],
    *,
    image_count: int,
    dataset_manifest: Mapping[str, Any],
) -> None:
    """Require Berkeley package identity before enabling a formal claim.

    The structural checks in :func:`_validate_prepared_evidence` (image names,
    label categories, and frame cardinality) intentionally also support small
    synthetic fixtures.  They are not sufficient to distinguish a community
    ``det_val.json`` from Berkeley's Detection 2020 package.  For the real
    10,000-image lane, require the two published package MD5 attestations and
    the Berkeley portal provenance.  The direct HTTP mirror is intentionally
    *not* allow-listed: it has returned the wrong payload in the past, and a
    URL alone is never accepted as content identity.

    This function validates receipt metadata only; preparation remains the
    place where the MD5 values are computed over the operator's local files.
    """

    if image_count != BDD100K_OFFICIAL_IMAGE_COUNT:
        return

    if source_receipt.get("schema_version") != "roadsense.bdd100k-detection-source-receipt/v1":
        raise BDD100KFinalizeError("formal BDD100K source receipt schema is unsupported")
    dataset_source_url = dataset_manifest.get("source_url")
    if not _is_official_berkeley_url(dataset_source_url):
        raise BDD100KFinalizeError(
            "formal BDD100K dataset manifest must cite the Berkeley source portal"
        )

    archives = source_receipt.get("source_archives")
    if not isinstance(archives, list) or len(archives) != len(BDD100K_OFFICIAL_ARCHIVE_ROLES):
        raise BDD100KFinalizeError(
            "formal BDD100K source receipt must contain exactly the images and "
            "det_20 labels packages"
        )
    by_role: dict[str, Mapping[str, Any]] = {}
    for index, archive in enumerate(archives):
        if not isinstance(archive, Mapping):
            raise BDD100KFinalizeError(f"source_archives[{index}] must be an object")
        role = archive.get("role")
        if not isinstance(role, str) or role in by_role:
            raise BDD100KFinalizeError("formal BDD100K source archive roles are invalid")
        by_role[role] = archive
    if set(by_role) != set(BDD100K_OFFICIAL_ARCHIVE_ROLES):
        raise BDD100KFinalizeError(
            "formal BDD100K source receipt must include images_val_zip and det_20_labels"
        )

    expected_md5 = {
        "images_val_zip": BDD100K_OFFICIAL_IMAGES_MD5,
        "det_20_labels": BDD100K_OFFICIAL_LABELS_MD5,
    }
    for role, expected_format in BDD100K_OFFICIAL_ARCHIVE_ROLES.items():
        archive = by_role[role]
        if archive.get("format") != expected_format:
            raise BDD100KFinalizeError(
                f"formal BDD100K {role} archive must be the official ZIP package"
            )
        observed_md5 = archive.get("official_package_md5")
        if not isinstance(observed_md5, str) or observed_md5.lower() != expected_md5[role]:
            raise BDD100KFinalizeError(
                f"formal BDD100K {role} archive is missing the published official MD5"
            )
        # ``official_source_url`` is retained for local provenance, but it
        # must point at the Berkeley portal rather than the stale/mutable raw
        # mirror.  Content identity still comes from the MD5 above.
        source_url = archive.get("official_source_url")
        if not _is_official_berkeley_url(source_url):
            raise BDD100KFinalizeError(
                f"formal BDD100K {role} archive must cite the Berkeley source portal"
            )


def _expected_content_sha256(
    *, image_count: int, images_tree_sha256: str, labels_sha256: str
) -> str:
    return canonical_sha256(
        {
            "dataset_id": "BDD100K",
            "task": "detection",
            "split": "val",
            "images_tree_sha256": images_tree_sha256,
            "labels_sha256": labels_sha256,
            "image_count": image_count,
        }
    )


def _validate_prepared_evidence(
    *,
    source_receipt: Mapping[str, Any],
    dataset_manifest: Mapping[str, Any],
    image_manifest: Mapping[str, Any],
    split_inventory: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if (
        source_receipt.get("dataset_id") != "BDD100K"
        or source_receipt.get("task") != "detection"
        or source_receipt.get("split") != "val"
    ):
        raise BDD100KFinalizeError("source receipt is not BDD100K Detection val")
    if source_receipt.get("local_only") is not True:
        raise BDD100KFinalizeError("source receipt must be local-only evidence")
    acceptance = source_receipt.get("license_acceptance")
    if not isinstance(acceptance, Mapping) or acceptance.get("accepted") is not True:
        raise BDD100KFinalizeError("BDD100K research-license acceptance is missing")
    if dataset_manifest.get("schema_version") != "roadsense.dataset-manifest/v1":
        raise BDD100KFinalizeError("unexpected dataset manifest schema")
    if dataset_manifest.get("dataset_name") != "BDD100K Detection 2020 validation":
        raise BDD100KFinalizeError("dataset manifest is not BDD100K Detection 2020 val")
    if dataset_manifest.get("evaluation_authorized") is not True:
        raise BDD100KFinalizeError("dataset manifest is not evaluation-authorized")
    if not isinstance(dataset_manifest.get("tasks"), list) or dataset_manifest["tasks"] != [
        "detection"
    ]:
        raise BDD100KFinalizeError("dataset manifest task must be detection")

    frozen = image_manifest
    if (
        frozen.get("dataset_id") != "BDD100K"
        or frozen.get("task") != "detection"
        or frozen.get("split") != "val"
    ):
        raise BDD100KFinalizeError("image manifest is not BDD100K Detection val")
    if frozen.get("frozen") is not True:
        raise BDD100KFinalizeError("finalization requires a frozen image manifest")
    try:
        # Validate the self-binding digest and image tree without opening labels.
        validated_image_manifest = load_frozen_image_manifest_from_mapping(frozen)
    except FrozenManifestError as exc:
        raise BDD100KFinalizeError(str(exc)) from exc
    image_count = validated_image_manifest["image_count"]
    images_tree_sha256 = _require_sha(
        validated_image_manifest["images_tree_sha256"], "images_tree_sha256"
    )
    _validate_official_source_attestation(
        source_receipt,
        image_count=int(image_count),
        dataset_manifest=dataset_manifest,
    )
    split_manifest_sha256 = _require_sha(
        validated_image_manifest["manifest_sha256"], "manifest_sha256"
    )
    declared_source_tree = _require_sha(
        source_receipt.get("images_tree_sha256"), "source_receipt.images_tree_sha256"
    )
    if declared_source_tree != images_tree_sha256:
        raise BDD100KFinalizeError("source receipt image tree hash disagrees with frozen manifest")

    archives = _archive_material(source_receipt)
    source_archives = source_receipt.get("source_archives")
    if validated_image_manifest.get("source_archives") != source_archives:
        raise BDD100KFinalizeError(
            "frozen image manifest source archives disagree with source receipt"
        )

    labels_sha256 = source_receipt.get("labels_sha256")
    if split_inventory is not None:
        if (
            split_inventory.get("schema_version")
            != "roadsense.bdd100k-detection-split-inventory/v1"
        ):
            raise BDD100KFinalizeError("unexpected split inventory schema")
        labels = split_inventory.get("labels")
        if not isinstance(labels, Mapping):
            raise BDD100KFinalizeError("split inventory labels are missing")
        if split_inventory.get("dataset_id") != "BDD100K":
            raise BDD100KFinalizeError("split inventory dataset is not BDD100K")
        if split_inventory.get("task") != "detection" or split_inventory.get("split") != "val":
            raise BDD100KFinalizeError("split inventory is not BDD100K Detection val")
        if split_inventory.get("image_count") != image_count:
            raise BDD100KFinalizeError("split inventory image count disagrees with frozen manifest")
        if split_inventory.get("images_tree_sha256") != images_tree_sha256:
            raise BDD100KFinalizeError(
                "split inventory image tree hash disagrees with frozen manifest"
            )
        inventory_hash = labels.get("sha256")
        if labels_sha256 is not None and inventory_hash != labels_sha256:
            raise BDD100KFinalizeError("source receipt and split inventory label hashes disagree")
        labels_sha256 = inventory_hash
        inventory_bytes = labels.get("bytes")
        if (
            isinstance(inventory_bytes, bool)
            or not isinstance(inventory_bytes, int)
            or inventory_bytes < 1
        ):
            raise BDD100KFinalizeError("split inventory label byte count is invalid")
        source_bytes = source_receipt.get("labels_bytes")
        if source_bytes is not None and source_bytes != inventory_bytes:
            raise BDD100KFinalizeError(
                "source receipt and split inventory label byte counts disagree"
            )
        frame_count = labels.get("frame_count")
        if frame_count != image_count:
            raise BDD100KFinalizeError(
                "split inventory label frame count disagrees with frozen manifest"
            )
    labels_sha256 = _require_sha(labels_sha256, "ground_truth_sha256")
    archive_md5_by_role = {
        str(item["role"]): item.get("official_package_md5")
        for item in source_archives
        if isinstance(item.get("role"), str)
    }
    images_package_md5 = archive_md5_by_role.get("images_val_zip")
    labels_package_md5 = archive_md5_by_role.get("det_20_labels")
    if image_count == BDD100K_OFFICIAL_IMAGE_COUNT and (
        not isinstance(images_package_md5, str) or not isinstance(labels_package_md5, str)
    ):
        raise BDD100KFinalizeError("formal BDD100K package MD5 attestations are missing")
    expected_content = _expected_content_sha256(
        image_count=image_count, images_tree_sha256=images_tree_sha256, labels_sha256=labels_sha256
    )
    for owner, payload in (
        ("source receipt", source_receipt),
        ("dataset manifest", dataset_manifest),
    ):
        declared = payload.get("content_sha256")
        if declared is not None and declared != expected_content:
            raise BDD100KFinalizeError(f"{owner} content_sha256 does not bind prepared evidence")
    return {
        "archive_sha256": canonical_sha256(archives),
        "tree_sha256": images_tree_sha256,
        "ground_truth_sha256": labels_sha256,
        "split_manifest_sha256": split_manifest_sha256,
        "image_count": image_count,
        "images_package_md5": images_package_md5,
        "labels_package_md5": labels_package_md5,
        "content_sha256": expected_content,
    }


def load_frozen_image_manifest_from_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate an in-memory frozen manifest without relying on a file path."""

    # The public validator only checks the hash-bound image inventory; it never
    # opens labels or media.  Import it at module load with the same direct-
    # script fallback as the CLI so this helper works in both invocation modes.
    return validate_frozen_image_manifest(value)


def _validate_model_and_inference(
    *,
    model_manifest: Mapping[str, Any],
    inference_receipt: Mapping[str, Any],
    image_evidence: Mapping[str, Any],
    model_manifest_base: Path | None,
    inference_base: Path | None,
) -> dict[str, Any]:
    if model_manifest.get("schema_version") != "roadsense.bdd100k-detection-model/v1":
        raise BDD100KFinalizeError("unexpected model manifest schema")
    model_id = _require_identifier(model_manifest.get("model_id"), "model_id")
    artifact_sha = _require_sha(model_manifest.get("artifact_sha256"), "artifact_sha256")
    artifact_bytes = model_manifest.get("artifact_bytes")
    if (
        isinstance(artifact_bytes, bool)
        or not isinstance(artifact_bytes, int)
        or artifact_bytes < 1
    ):
        raise BDD100KFinalizeError("model artifact_bytes must be positive")
    ontology = model_manifest.get("ontology")
    if not isinstance(ontology, Mapping) or not isinstance(ontology.get("mapping"), Mapping):
        raise BDD100KFinalizeError("model manifest ontology mapping is missing")
    ontology_map_sha = canonical_sha256(ontology["mapping"])
    # The inference receipt records the canonical model-manifest digest.  Bind
    # it to the object supplied to this finalizer instead of trusting a hash
    # copied from an unrelated run.
    model_manifest_sha = canonical_sha256(model_manifest)
    if (
        inference_receipt.get("schema_version") != "roadsense.bdd100k-detection-inference/v1"
        or inference_receipt.get("stage") != "infer"
    ):
        raise BDD100KFinalizeError("unexpected inference receipt schema or stage")
    model = inference_receipt.get("model")
    if not isinstance(model, Mapping):
        raise BDD100KFinalizeError("inference receipt model provenance is missing")
    if model.get("sha256") != artifact_sha:
        raise BDD100KFinalizeError("inference model hash disagrees with model manifest")
    declared_model_manifest_sha = _require_sha(
        model.get("manifest_sha256"), "model.manifest_sha256"
    )
    if declared_model_manifest_sha != model_manifest_sha:
        raise BDD100KFinalizeError(
            "inference model manifest hash does not match the supplied model manifest"
        )
    inference_dataset = inference_receipt.get("dataset")
    if not isinstance(inference_dataset, Mapping):
        raise BDD100KFinalizeError("inference dataset provenance is missing")
    for field, expected in (
        ("image_manifest_sha256", image_evidence.get("manifest_sha256")),
        ("images_tree_sha256", image_evidence.get("images_tree_sha256")),
        ("image_count", image_evidence.get("image_count")),
    ):
        observed = inference_dataset.get(field)
        if observed != expected:
            raise BDD100KFinalizeError(
                f"inference dataset {field} does not match the frozen image manifest"
            )
    inference_ontology = inference_receipt.get("ontology")
    if (
        not isinstance(inference_ontology, Mapping)
        or inference_ontology.get("mapping_sha256") != ontology_map_sha
    ):
        raise BDD100KFinalizeError("inference ontology map does not match model manifest")
    inference = inference_receipt.get("inference")
    prediction = inference_receipt.get("prediction")
    if not isinstance(inference, Mapping) or not isinstance(prediction, Mapping):
        raise BDD100KFinalizeError("inference config/prediction provenance is missing")
    config_sha = _require_sha(inference.get("config_sha256"), "inference.config_sha256")
    prediction_sha = _require_sha(prediction.get("sha256"), "prediction.sha256")
    _check_optional_file_hash(
        base=inference_base,
        relative_name=prediction.get("path"),
        expected_sha=prediction_sha,
        field="prediction",
    )
    # Bind the canonical model-manifest content (rather than its formatting
    # bytes) to the inference receipt.  This closes the provenance link even
    # when the manifest is supplied as an in-memory synthetic object.
    if model_manifest_sha != canonical_sha256(model_manifest):
        raise BDD100KFinalizeError(
            "inference model manifest hash does not bind model-manifest.json"
        )
    return {
        "model_id": model_id,
        "artifact_sha256": artifact_sha,
        "manifest_sha256": declared_model_manifest_sha,
        "ontology_map_sha256": ontology_map_sha,
        "config_sha256": config_sha,
        "prediction_sha256": prediction_sha,
    }


def _validate_evaluator_receipt(
    value: Mapping[str, Any],
    *,
    base: Path | None,
    expected_ground_truth_sha256: str,
    expected_image_manifest_sha256: str,
    expected_prediction_sha256: str,
    expected_image_count: int,
) -> dict[str, Any]:
    if (
        value.get("schema_version") != "roadsense.bdd100k-detection-evaluation/v1"
        or value.get("stage") != "evaluate"
    ):
        raise BDD100KFinalizeError("unexpected evaluation receipt schema or stage")
    if value.get("status") != "ok":
        raise BDD100KFinalizeError("both evaluator receipts must have status=ok")
    prediction = value.get("prediction")
    evaluator = value.get("evaluator")
    if not isinstance(prediction, Mapping) or not isinstance(evaluator, Mapping):
        raise BDD100KFinalizeError("evaluation receipt provenance is incomplete")
    prediction_sha = _require_sha(prediction.get("sha256"), "evaluation.prediction.sha256")
    dataset = value.get("dataset")
    if not isinstance(dataset, Mapping):
        raise BDD100KFinalizeError("evaluation dataset provenance is missing")
    ground_truth_sha = _require_sha(
        dataset.get("ground_truth_sha256"), "evaluation.ground_truth_sha256"
    )
    if ground_truth_sha != expected_ground_truth_sha256:
        raise BDD100KFinalizeError(
            "evaluation receipt ground truth hash does not match prepared data"
        )
    if dataset.get("split") != "val":
        raise BDD100KFinalizeError("evaluation receipt split must be val")
    if dataset.get("image_manifest_sha256") != expected_image_manifest_sha256:
        raise BDD100KFinalizeError(
            "evaluation receipt image manifest hash does not match the frozen manifest"
        )
    frame_count = prediction.get("frame_count")
    if (
        isinstance(frame_count, bool)
        or not isinstance(frame_count, int)
        or frame_count != expected_image_count
    ):
        raise BDD100KFinalizeError("evaluation prediction frame count does not match the split")
    if prediction_sha != expected_prediction_sha256:
        raise BDD100KFinalizeError("evaluation prediction hash does not match inference output")
    result_sha = _require_sha(evaluator.get("result_sha256"), "evaluation.result_sha256")
    nested_result = evaluator.get("result")
    if isinstance(nested_result, Mapping) and nested_result.get("sha256") != result_sha:
        raise BDD100KFinalizeError("evaluation result hash fields disagree")
    metrics = _finite_metrics(evaluator.get("metrics"), "evaluation.metrics")
    commit = evaluator.get("commit")
    if commit != BDD100K_DEVKIT_COMMIT:
        raise BDD100KFinalizeError("evaluation receipt uses an unpinned BDD devkit commit")
    if evaluator.get("id") != BDD100K_DEVKIT_ID or evaluator.get("module") != "bdd100k.eval.run":
        raise BDD100KFinalizeError("evaluation receipt evaluator identity is unsupported")
    config_sha = _require_sha(evaluator.get("evaluator_config_sha256"), "evaluator_config_sha256")
    lock_sha = _require_sha(evaluator.get("runtime_lock_sha256"), "evaluator.runtime_lock_sha256")
    packages = evaluator.get("packages")
    if not isinstance(packages, Mapping) or not packages:
        raise BDD100KFinalizeError("evaluator dependency package lock is missing")
    normalized_packages: dict[str, str] = {}
    for name, version in packages.items():
        if not isinstance(name, str) or not isinstance(version, str) or not name or not version:
            raise BDD100KFinalizeError("evaluator dependency lock contains invalid values")
        if any(char in version for char in ("/", "\\", ":", "@")):
            raise BDD100KFinalizeError("evaluator dependency lock exposes a path")
        normalized_packages[name] = version
    if expected_image_count == BDD100K_OFFICIAL_IMAGE_COUNT:
        if normalized_packages != BDD100K_REQUIRED_EVALUATOR_PACKAGES:
            raise BDD100KFinalizeError(
                "formal BDD100K evaluator packages must exactly match the validated lock"
            )
        # ``status=ok`` is generated by the runner only after a zero exit and
        # parseable result.  Requiring the underlying fields here prevents a
        # hand-edited receipt from converting a failed wrapper into a claim.
        if evaluator.get("returncode") != 0 or evaluator.get("timed_out") is not False:
            raise BDD100KFinalizeError(
                "formal BDD100K evaluator receipt must record returncode=0 and timed_out=false"
            )
        result_source = evaluator.get("result_source")
        if result_source != "file":
            raise BDD100KFinalizeError(
                "formal BDD100K evaluator receipt must bind a result file (not stdout fallback)"
            )
    run_id = _require_identifier(value.get("run_id"), "evaluation.run_id")
    result_path = _check_optional_file_hash(
        base=base,
        relative_name=evaluator.get("result", {}).get("path")
        if isinstance(evaluator.get("result"), Mapping)
        else None,
        expected_sha=result_sha,
        field="evaluator result",
    )
    if expected_image_count == BDD100K_OFFICIAL_IMAGE_COUNT:
        if result_path is None:
            raise BDD100KFinalizeError(
                "formal BDD100K evaluator receipt requires a path-backed result artifact"
            )
        bound_metrics = _load_bound_official_detection_metrics(
            result_path, expected_sha256=result_sha
        )
        if metrics != bound_metrics:
            raise BDD100KFinalizeError(
                "evaluation receipt metrics do not match the bound official evaluator result"
            )
        metrics = bound_metrics
    stdout_sha = _require_sha(evaluator.get("stdout_sha256"), "evaluator.stdout_sha256")
    stderr_sha = _require_sha(evaluator.get("stderr_sha256"), "evaluator.stderr_sha256")
    if base is not None:
        for filename, expected, field in (
            ("evaluator.stdout.txt", stdout_sha, "evaluator stdout"),
            ("evaluator.stderr.txt", stderr_sha, "evaluator stderr"),
        ):
            path = (base / filename).resolve()
            if not path.is_file() or path.is_symlink():
                raise BDD100KFinalizeError(f"{field} artifact is missing")
            observed, _ = _sha256_file(path)
            if observed != expected:
                raise BDD100KFinalizeError(f"{field} hash does not match receipt")
    role = value.get("role")
    if role is not None and role not in ("independent_a", "independent_b"):
        raise BDD100KFinalizeError("evaluation role must be independent_a or independent_b")
    return {
        "role": role,
        "run_id": run_id,
        "prediction_sha256": prediction_sha,
        "ground_truth_sha256": ground_truth_sha,
        "result_sha256": result_sha,
        "metrics": metrics,
        "config_sha256": config_sha,
        "lock_sha256": lock_sha,
        "packages": dict(sorted(normalized_packages.items())),
        "stdout_sha256": stdout_sha,
        "stderr_sha256": stderr_sha,
    }


def finalize_bdd100k_detection_benchmark(
    *,
    source_receipt: Path | str | Mapping[str, Any],
    dataset_manifest: Path | str | Mapping[str, Any],
    image_manifest: Path | str | Mapping[str, Any],
    model_manifest: Path | str | Mapping[str, Any],
    inference_receipt: Path | str | Mapping[str, Any],
    evaluation_a: Path | str | Mapping[str, Any],
    evaluation_b: Path | str | Mapping[str, Any],
    output: Path | str | None = None,
    split_inventory: Path | str | Mapping[str, Any],
) -> dict[str, Any]:
    """Validate all frozen evidence and emit a sanitized benchmark receipt."""

    source, _source_base = _load_source(source_receipt, role="source receipt")
    dataset, _dataset_base = _load_source(dataset_manifest, role="dataset manifest")
    image, _image_base = _load_source(image_manifest, role="image manifest")
    model, model_base = _load_source(model_manifest, role="model manifest")
    inference, inference_base = _load_source(inference_receipt, role="inference receipt")
    eval_a, eval_a_base = _load_source(evaluation_a, role="evaluation receipt A")
    eval_b, eval_b_base = _load_source(evaluation_b, role="evaluation receipt B")
    split, _split_base = _load_source(split_inventory, role="split inventory")

    dataset_evidence = _validate_prepared_evidence(
        source_receipt=source,
        dataset_manifest=dataset,
        image_manifest=image,
        split_inventory=split,
    )
    model_evidence = _validate_model_and_inference(
        model_manifest=model,
        inference_receipt=inference,
        image_evidence=image,
        model_manifest_base=model_base,
        inference_base=inference_base,
    )
    inference_dataset = inference.get("dataset")
    if not isinstance(inference_dataset, Mapping):
        raise BDD100KFinalizeError("inference dataset provenance is missing")
    if (
        inference_dataset.get("image_manifest_sha256") != image["manifest_sha256"]
        or inference_dataset.get("images_tree_sha256") != dataset_evidence["tree_sha256"]
        or inference_dataset.get("image_count") != dataset_evidence["image_count"]
    ):
        raise BDD100KFinalizeError("inference receipt does not bind the frozen image manifest")
    first = _validate_evaluator_receipt(
        eval_a,
        base=eval_a_base,
        expected_ground_truth_sha256=cast(str, dataset_evidence["ground_truth_sha256"]),
        expected_image_manifest_sha256=cast(str, image["manifest_sha256"]),
        expected_prediction_sha256=cast(str, model_evidence["prediction_sha256"]),
        expected_image_count=int(dataset_evidence["image_count"]),
    )
    second = _validate_evaluator_receipt(
        eval_b,
        base=eval_b_base,
        expected_ground_truth_sha256=cast(str, dataset_evidence["ground_truth_sha256"]),
        expected_image_manifest_sha256=cast(str, image["manifest_sha256"]),
        expected_prediction_sha256=cast(str, model_evidence["prediction_sha256"]),
        expected_image_count=int(dataset_evidence["image_count"]),
    )
    if first["run_id"] == second["run_id"]:
        raise BDD100KFinalizeError("evaluator runs must have distinct run_id values")
    if (
        first["prediction_sha256"] != model_evidence["prediction_sha256"]
        or second["prediction_sha256"] != model_evidence["prediction_sha256"]
    ):
        raise BDD100KFinalizeError("both evaluator runs must bind the inference prediction hash")
    if (
        first["ground_truth_sha256"] != dataset_evidence["ground_truth_sha256"]
        or second["ground_truth_sha256"] != dataset_evidence["ground_truth_sha256"]
    ):
        raise BDD100KFinalizeError("both evaluator runs must bind the prepared BDD ground truth")
    if first["metrics"] != second["metrics"]:
        raise BDD100KFinalizeError("independent evaluator runs must report identical metrics")
    if first["config_sha256"] != second["config_sha256"]:
        raise BDD100KFinalizeError("independent evaluator runs use different evaluator configs")
    if first["lock_sha256"] != second["lock_sha256"] or first["packages"] != second["packages"]:
        raise BDD100KFinalizeError("independent evaluator runs use different dependency locks")
    # Validate reduced fixtures deeply enough to exercise integrity errors, but
    # never serialize one as a public BDD100K benchmark receipt.
    if dataset_evidence["image_count"] != BDD100K_OFFICIAL_IMAGE_COUNT:
        raise BDD100KFinalizeError(
            "formal BDD100K benchmark requires exactly 10000 validation images"
        )
    explicit_roles = [item["role"] for item in (first, second) if item["role"] is not None]
    if dataset_evidence["image_count"] == BDD100K_OFFICIAL_IMAGE_COUNT and len(explicit_roles) != 2:
        raise BDD100KFinalizeError(
            "formal BDD100K evaluator receipts must explicitly declare independent_a and "
            "independent_b roles"
        )
    if explicit_roles and set(explicit_roles) != {"independent_a", "independent_b"}:
        raise BDD100KFinalizeError(
            "explicit evaluator roles must be independent_a and independent_b"
        )
    roles = {first["role"], second["role"]}
    if roles == {"independent_a", "independent_b"}:
        ordered = sorted((first, second), key=lambda item: item["role"])
    else:
        # Older runner receipts did not carry a role.  Canonicalize by run id
        # and assign roles deterministically, so argument order cannot alter a
        # published report identifier.
        ordered = sorted((first, second), key=lambda item: item["run_id"])
        ordered[0]["role"], ordered[1]["role"] = "independent_a", "independent_b"

    evaluator_payload = {
        "evaluator_id": BDD100K_DEVKIT_ID,
        "repository": BDD100K_DEVKIT_REPOSITORY,
        "commit": BDD100K_DEVKIT_COMMIT,
        "config_sha256": ordered[0]["config_sha256"],
        "dependencies": {
            "lock_sha256": ordered[0]["lock_sha256"],
            "packages": ordered[0]["packages"],
        },
    }
    model_payload = {
        "model_id": model_evidence["model_id"],
        "artifact_sha256": model_evidence["artifact_sha256"],
        "manifest_sha256": model_evidence["manifest_sha256"],
        "ontology_map_sha256": model_evidence["ontology_map_sha256"],
    }
    inference_payload = {
        "config_sha256": model_evidence["config_sha256"],
        "prediction_sha256": model_evidence["prediction_sha256"],
    }
    payload: dict[str, Any] = {
        "schema_version": BDD100K_DETECTION_BENCHMARK_SCHEMA,
        "scope": BDD100K_DETECTION_SCOPE,
        "evidence_level": "frozen_evaluation",
        "evaluation_authorized": True,
        "frozen": True,
        "benchmark_claim_available": True,
        "dataset": {
            "dataset": "BDD100K",
            "task": "detection",
            "release": "2020",
            "split": "val",
            "image_count": dataset_evidence["image_count"],
            "images_package_md5": dataset_evidence["images_package_md5"],
            "labels_package_md5": dataset_evidence["labels_package_md5"],
            **{
                key: dataset_evidence[key]
                for key in (
                    "archive_sha256",
                    "tree_sha256",
                    "ground_truth_sha256",
                    "split_manifest_sha256",
                )
            },
        },
        "model": model_payload,
        "inference": inference_payload,
        "evaluator": evaluator_payload,
        "evaluator_runs": [
            {
                "role": item["role"],
                "run_id": item["run_id"],
                "output_sha256": item["result_sha256"],
                "metrics": item["metrics"],
            }
            for item in ordered
        ],
    }
    receipt = build_bdd100k_detection_receipt(payload).model_dump(mode="json")
    if output is not None:
        write_json_atomic(Path(output), receipt)
    return receipt


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for flag, dest in (
        ("--source-receipt", "source_receipt"),
        ("--dataset-manifest", "dataset_manifest"),
        ("--image-manifest", "image_manifest"),
        ("--model-manifest", "model_manifest"),
        ("--inference-receipt", "inference_receipt"),
        ("--evaluation-a", "evaluation_a"),
        ("--evaluation-b", "evaluation_b"),
    ):
        parser.add_argument(flag, dest=dest, type=Path, required=True)
    parser.add_argument("--split-inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        receipt = finalize_bdd100k_detection_benchmark(
            source_receipt=args.source_receipt,
            dataset_manifest=args.dataset_manifest,
            image_manifest=args.image_manifest,
            model_manifest=args.model_manifest,
            inference_receipt=args.inference_receipt,
            evaluation_a=args.evaluation_a,
            evaluation_b=args.evaluation_b,
            split_inventory=args.split_inventory,
            output=args.output,
        )
    except (
        BDD100KFinalizeError,
        BDD100KBenchmarkReceiptError,
        FrozenManifestError,
        OSError,
        TypeError,
        ValueError,
        MemoryError,
    ) as exc:
        print(f"BDD100K benchmark finalization failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"status": "ok", "report_id": receipt["report_id"]}, sort_keys=True))
    return 0


__all__ = [
    "FINALIZE_SCHEMA",
    "BDD100KFinalizeError",
    "compute_archive_sha256",
    "finalize_bdd100k_detection_benchmark",
    "load_frozen_image_manifest_from_mapping",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
