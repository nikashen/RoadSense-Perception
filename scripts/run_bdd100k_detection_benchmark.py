"""Auditable BDD100K Detection 2020 validation runner.

The runner is deliberately split into two processes/stages:

``freeze``
    Turns the image manifest produced by :mod:`prepare_bdd100k_detection` into
    an immutable, hash-bound manifest.  The operation only deals with image
    metadata; it never needs to open the BDD labels.

``infer``
    Runs a local YOLO11 ONNX graph against the frozen image manifest and writes
    a Scalabel/BDD prediction list.  Ground-truth files are not accepted by
    this stage.  Every manifest image receives a frame, including images for
    which the detector returns no supported class.

``evaluate``
    Invokes the pinned BDD100K devkit in an *external* Python interpreter::

        python -m bdd100k.eval.run -t det -g GT -r PRED

No download is performed here.  BDD100K packages, model weights, and the
devkit environment are operator-provided local inputs.  The receipts capture
hashes, commands, runtime locks, and evaluator streams so a later publication
can distinguish a real official-devkit result from a development fixture.

The image preprocessing and YOLO tensor decoder are shared with the COCO8
runner instead of being copied.  The decoder still operates in the 80-class
COCO ontology; this module applies an explicit, intentionally partial COCO to
BDD mapping afterwards.  ``rider`` and ``traffic sign`` have no faithful
one-to-one COCO classes and are never silently fabricated.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np

from roadsense.bdd_benchmark import BDD100K_REQUIRED_EVALUATOR_PACKAGES
from roadsense.json_io import canonical_sha256, load_strict_json, write_json_atomic

# Reuse the already-tested YOLO11 implementation.  Keeping these imports
# private makes it obvious that the BDD-specific part is the ontology and
# protocol adapter, not a second subtly different tensor decoder.
try:
    # Import works when the repository root is on ``sys.path`` (pytest,
    # editable installs, and ``python -m``).
    from scripts.run_coco8_onnx_eval import (
        COCO_CLASS_NAMES,
        INPUT_SIZE,
        MODEL_SOURCE_URL,
    )
    from scripts.run_coco8_onnx_eval import (
        _decode_predictions as _coco_decode_predictions,
    )
    from scripts.run_coco8_onnx_eval import (
        _letterbox as _coco_letterbox,
    )
    from scripts.run_coco8_onnx_eval import (
        _validate_onnx_session as _coco_validate_onnx_session,
    )
except ModuleNotFoundError:  # pragma: no cover - exercised by direct script CLI
    # ``python scripts/run_bdd100k_detection_benchmark.py`` puts only the
    # scripts directory on sys.path, so use the sibling module fallback.
    from run_coco8_onnx_eval import (  # type: ignore[no-redef]
        COCO_CLASS_NAMES,
        INPUT_SIZE,
        MODEL_SOURCE_URL,
    )
    from run_coco8_onnx_eval import (
        _decode_predictions as _coco_decode_predictions,
    )
    from run_coco8_onnx_eval import (
        _letterbox as _coco_letterbox,
    )
    from run_coco8_onnx_eval import (
        _validate_onnx_session as _coco_validate_onnx_session,
    )

BDD100K_DETECTION_CATEGORIES = (
    "pedestrian",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
    "traffic light",
    "traffic sign",
)
BDD100K_CATEGORIES = BDD100K_DETECTION_CATEGORIES

# Only semantically defensible mappings are included.  In particular,
# COCO's ``stop sign`` is not promoted to BDD's broader ``traffic sign`` and a
# generic person is not guessed to be a rider.  Missing BDD classes are exposed
# in every inference receipt.
# Keep an entry for every COCO class.  ``None`` is deliberate evidence that a
# class has no BDD equivalent; omitting it would make an accidental fallback
# or silent ontology drift harder to detect in a receipt.
_COCO_TO_BDD_SUPPORTED: dict[int, str] = {
    0: "pedestrian",  # person
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    6: "train",
    7: "truck",
    9: "traffic light",
}
COCO_TO_BDD_CATEGORY: dict[int, str | None] = {
    index: _COCO_TO_BDD_SUPPORTED.get(index) for index in range(len(COCO_CLASS_NAMES))
}
COCO_TO_BDD: dict[str, str | None] = {
    COCO_CLASS_NAMES[index]: category for index, category in COCO_TO_BDD_CATEGORY.items()
}
UNSUPPORTED_BDD_CATEGORIES = ("rider", "traffic sign")

IMAGE_MANIFEST_SCHEMA = "roadsense.bdd100k-detection-images/v1"
FROZEN_IMAGE_MANIFEST_SCHEMA = "roadsense.bdd100k-detection-frozen-images/v1"
PREDICTION_SCHEMA = "scalabel.bdd100k-detection/v1"
INFERENCE_RECEIPT_SCHEMA = "roadsense.bdd100k-detection-inference/v1"
EVALUATION_RECEIPT_SCHEMA = "roadsense.bdd100k-detection-evaluation/v1"
DEVKIT_MODULE = "bdd100k.eval.run"
DEVKIT_ID = "bdd100k-devkit"
DEVKIT_COMMIT = "9ac17c6c7c51d2fc83065fccd707cd5b1882a293"
# The pinned bdd100k/scalabel evaluator path is incompatible with the tested
# pycocotools 2.0.9/2.0.10 releases (they raise ``KeyError: info`` while
# loading detection results).  Keep the known-good version in the runtime
# gate and in every evaluator receipt rather than relying on an operator's
# ambient env.
REQUIRED_PYCOCOTOOLS_VERSION = BDD100K_REQUIRED_EVALUATOR_PACKAGES["pycocotools"]
MAX_IMAGE_COUNT = 100_000


class BDDRunnerError(ValueError):
    """Raised when a local BDD runner input or evidence record is invalid."""


class FrozenManifestError(BDDRunnerError):
    """Raised when inference is attempted with a mutable/unbound manifest."""


class EvaluatorError(BDDRunnerError):
    """Raised when an isolated official-devkit invocation fails."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> tuple[str, int]:
    """Hash a file in bounded chunks and return ``(digest, byte_count)``."""

    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
                size += len(chunk)
    except OSError as exc:
        raise BDDRunnerError(f"cannot read local file: {path}") from exc
    return digest.hexdigest(), size


def _write_generated_bytes(path: Path, payload: bytes, *, field: str) -> None:
    """Write a runner-generated artifact without following an existing link."""

    if path.is_symlink():
        raise EvaluatorError(f"{field} path must not be a symlink")
    if path.exists() and not path.is_file():
        raise EvaluatorError(f"{field} path must be a regular file")
    try:
        path.write_bytes(payload)
    except OSError as exc:
        raise EvaluatorError(f"unable to write {field}") from exc


def _strict_sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise FrozenManifestError(f"{field} must be a lowercase SHA-256 digest")
    if value.lower() != value or any(character not in "0123456789abcdef" for character in value):
        raise FrozenManifestError(f"{field} must be a lowercase SHA-256 digest")
    if value == "0" * 64:
        raise FrozenManifestError(f"{field} cannot be all-zero")
    return value


def _safe_image_name(value: object) -> bool:
    if not isinstance(value, str) or not value or value != value.strip():
        return False
    if "/" in value or "\\" in value or ":" in value or "\x00" in value:
        return False
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        return False
    return value.lower().endswith((".jpg", ".jpeg", ".png"))


def _manifest_material(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return the hash material for a frozen manifest.

    ``manifest_sha256`` is a self-reference and therefore excluded.  Other
    fields, including the explicit ``frozen`` marker, remain bound.
    """

    return {key: item for key, item in value.items() if key != "manifest_sha256"}


def compute_image_manifest_sha256(value: Mapping[str, Any]) -> str:
    """Compute the canonical digest bound into a frozen image manifest."""

    return canonical_sha256(_manifest_material(value))


def _validate_image_manifest(value: object, *, require_frozen: bool) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise FrozenManifestError("image manifest must be a JSON object")
    raw = dict(value)
    if raw.get("schema_version") != IMAGE_MANIFEST_SCHEMA:
        raise FrozenManifestError(f"image manifest schema must be {IMAGE_MANIFEST_SCHEMA}")
    if raw.get("dataset_id") != "BDD100K" or raw.get("task") != "detection":
        raise FrozenManifestError("image manifest must describe BDD100K detection")
    if raw.get("split") != "val":
        raise FrozenManifestError("image manifest split must be val")
    image_count = raw.get("image_count")
    if isinstance(image_count, bool) or not isinstance(image_count, int):
        raise FrozenManifestError("image_count must be an integer")
    if not 1 <= image_count <= MAX_IMAGE_COUNT:
        raise FrozenManifestError("image_count is outside the supported range")
    images = raw.get("images")
    if not isinstance(images, list) or len(images) != image_count:
        raise FrozenManifestError("images must contain exactly image_count records")
    normalized_images: list[dict[str, Any]] = []
    names: list[str] = []
    for index, item in enumerate(images):
        if not isinstance(item, Mapping):
            raise FrozenManifestError(f"images[{index}] must be an object")
        name = item.get("name")
        if not _safe_image_name(name):
            raise FrozenManifestError(f"images[{index}].name is unsafe")
        assert isinstance(name, str)
        if name in names:
            raise FrozenManifestError("image manifest contains duplicate names")
        digest = _strict_sha256(item.get("sha256"), field=f"images[{index}].sha256")
        byte_count = item.get("bytes")
        if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count < 1:
            raise FrozenManifestError(f"images[{index}].bytes must be a positive integer")
        # Preserve only the protocol fields.  A source package may carry
        # additional local metadata, but it must not affect image identity.
        normalized_images.append({"name": name, "sha256": digest, "bytes": byte_count})
        names.append(name)
    if names != sorted(names):
        raise FrozenManifestError("image manifest images must be canonically sorted")
    tree_hash = _strict_sha256(raw.get("images_tree_sha256"), field="images_tree_sha256")
    if tree_hash != canonical_sha256(normalized_images):
        raise FrozenManifestError("images_tree_sha256 does not bind image records")

    frozen = raw.get("frozen", False)
    if not isinstance(frozen, bool):
        raise FrozenManifestError("frozen must be a boolean")
    if require_frozen and frozen is not True:
        raise FrozenManifestError(
            "inference accepts only a frozen image manifest (run the freeze stage first)"
        )
    if frozen:
        if raw.get("freeze_schema_version") != FROZEN_IMAGE_MANIFEST_SCHEMA:
            raise FrozenManifestError("frozen manifest has an unsupported freeze schema")
        bound = _strict_sha256(raw.get("manifest_sha256"), field="manifest_sha256")
        if bound != compute_image_manifest_sha256(raw):
            raise FrozenManifestError("manifest_sha256 does not bind the complete manifest")

    # Keep source archive metadata when present, but reject path-bearing image
    # records.  This output is later safe to copy into a publication receipt.
    normalized: dict[str, Any] = dict(raw)
    normalized["images"] = normalized_images
    normalized["images_tree_sha256"] = tree_hash
    normalized["image_count"] = image_count
    return normalized


def freeze_image_manifest(
    source: Path | str | Mapping[str, Any], destination: Path | str | None = None
) -> dict[str, Any]:
    """Create a canonical hash-bound frozen manifest.

    The source may be a path or an in-memory preparation manifest.  No label
    file is opened.  If ``destination`` is supplied, the frozen JSON is
    written atomically and the normalized payload is returned.
    """

    if isinstance(source, Mapping):
        raw: object = source
    else:
        try:
            raw = load_strict_json(Path(source))
        except (OSError, UnicodeError, ValueError, TypeError) as exc:
            raise FrozenManifestError("unable to load image manifest") from exc
    base = _validate_image_manifest(raw, require_frozen=False)
    # Re-freezing an already frozen manifest is allowed only when its binding
    # is valid; the validator above already checked it.
    base["frozen"] = True
    base["freeze_schema_version"] = FROZEN_IMAGE_MANIFEST_SCHEMA
    base.pop("manifest_sha256", None)
    base["manifest_sha256"] = compute_image_manifest_sha256(base)
    result = _validate_image_manifest(base, require_frozen=True)
    if destination is not None:
        write_json_atomic(Path(destination), result)
    return result


def load_frozen_image_manifest(path: Path | str) -> dict[str, Any]:
    """Strictly load a frozen manifest and verify its self-binding digest."""

    try:
        value = load_strict_json(Path(path))
    except (OSError, UnicodeError, ValueError, TypeError) as exc:
        raise FrozenManifestError(f"unable to load frozen image manifest: {path}") from exc
    return _validate_image_manifest(value, require_frozen=True)


# Friendly aliases used by downstream scripts and tests.
_load_frozen_image_manifest = load_frozen_image_manifest


def validate_frozen_image_manifest(value: object) -> dict[str, Any]:
    """Validate an in-memory frozen image manifest without opening labels."""

    return _validate_image_manifest(value, require_frozen=True)


def _letterbox(path: Path) -> tuple[np.ndarray, float, int, int, int, int]:
    """Shared YOLO11 letterbox adapter (kept as a BDD-local test seam)."""

    return _coco_letterbox(path)


def _decode_predictions(
    output: np.ndarray,
    *,
    scale: float,
    pad_x: int,
    pad_y: int,
    width: int,
    height: int,
    score_threshold: float,
    nms_iou: float,
    max_detections: int,
) -> tuple[Any, ...]:
    """Shared COCO YOLO decoder; categories are mapped in a later step."""

    return _coco_decode_predictions(
        output,
        scale=scale,
        pad_x=pad_x,
        pad_y=pad_y,
        width=width,
        height=height,
        score_threshold=score_threshold,
        nms_iou=nms_iou,
        max_detections=max_detections,
    )


_decode_yolo11_predictions = _decode_predictions


def map_coco_category(category_id: int) -> str | None:
    """Map a COCO class id to BDD, returning ``None`` for non-representable ids."""

    if isinstance(category_id, bool) or not isinstance(category_id, int):
        raise TypeError("COCO category id must be an integer")
    if not 0 <= category_id < len(COCO_CLASS_NAMES):
        raise ValueError("COCO category id is outside the 80-class ontology")
    return COCO_TO_BDD_CATEGORY[category_id]


def _prediction_label(category: str, score: float, box: Any, index: int) -> dict[str, Any]:
    if category not in BDD100K_DETECTION_CATEGORIES:
        raise BDDRunnerError(f"unsupported BDD category: {category}")
    if not math.isfinite(float(score)) or not 0.0 <= float(score) <= 1.0:
        raise BDDRunnerError("prediction score must be finite and in [0, 1]")
    # Detection.bbox is a roadsense BoxXYXY, but accepting a small mapping here
    # keeps this pure function useful in tests and in adapter integrations.
    if hasattr(box, "x_min"):
        x1, y1, x2, y2 = (float(box.x_min), float(box.y_min), float(box.x_max), float(box.y_max))
    elif isinstance(box, Mapping):
        x1 = float(box["x_min"])
        y1 = float(box["y_min"])
        x2 = float(box["x_max"])
        y2 = float(box["y_max"])
    else:
        raise BDDRunnerError("prediction box must expose x_min/y_min/x_max/y_max")
    values = (x1, y1, x2, y2)
    if not all(math.isfinite(item) for item in values) or x2 <= x1 or y2 <= y1:
        raise BDDRunnerError("prediction box must be finite and non-empty")
    return {
        "id": f"det-{index:06d}",
        "category": category,
        "score": float(score),
        "box2d": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        "attributes": {},
    }


def detections_to_bdd_labels(
    detections: Sequence[Any],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Convert COCO detections and explicitly count non-representable classes."""

    labels: list[dict[str, Any]] = []
    dropped: Counter[str] = Counter()
    for detection in detections:
        category_id = int(detection.category_id)
        category = map_coco_category(category_id)
        if category is None:
            dropped[COCO_CLASS_NAMES[category_id]] += 1
            continue
        labels.append(
            _prediction_label(category, float(detection.score), detection.bbox, len(labels))
        )
    # Decoder ordering is deterministic; sort once more to make the output
    # independent of a custom injected session's candidate ordering.
    labels.sort(key=lambda item: (-float(item["score"]), str(item["category"]), str(item["id"])))
    return labels, dict(sorted(dropped.items()))


def build_prediction_frame(
    image_name: str, detections: Sequence[Any]
) -> tuple[dict[str, Any], dict[str, int]]:
    """Build one Scalabel frame, including an explicit empty-label frame."""

    if not _safe_image_name(image_name):
        raise BDDRunnerError("prediction frame image name is unsafe")
    labels, dropped = detections_to_bdd_labels(detections)
    frame = {
        "name": image_name,
        "labels": labels,
        "attributes": {},
    }
    return frame, dropped


def build_prediction_document(
    image_names: Sequence[str], detections_by_image: Mapping[str, Sequence[Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build a complete, canonically ordered BDD prediction list.

    ``image_names`` is authoritative.  Extra detector keys are rejected and
    missing keys become empty frames, which prevents silently dropping images.
    """

    names = list(image_names)
    if names != sorted(names) or len(set(names)) != len(names):
        raise BDDRunnerError("image names must be unique and canonically sorted")
    extra = set(detections_by_image) - set(names)
    if extra:
        raise BDDRunnerError(
            f"detections contain images outside the frozen manifest: {sorted(extra)}"
        )
    frames: list[dict[str, Any]] = []
    dropped_total: Counter[str] = Counter()
    for name in names:
        frame, dropped = build_prediction_frame(name, detections_by_image.get(name, ()))
        frames.append(frame)
        dropped_total.update(dropped)
    metadata = {
        "mapped_coco_classes": dict(sorted(COCO_TO_BDD.items())),
        "unsupported_bdd_categories": list(UNSUPPORTED_BDD_CATEGORIES),
        "dropped_coco_detections": dict(sorted(dropped_total.items())),
        "frame_count": len(frames),
        "empty_frame_count": sum(not frame["labels"] for frame in frames),
    }
    return frames, metadata


def _resolve_image_root(dataset_root: Path, images_root: Path | None) -> Path:
    root = (
        (images_root if images_root is not None else dataset_root / "images" / "val")
        .expanduser()
        .resolve()
    )
    if not root.is_dir() or root.is_symlink():
        raise BDDRunnerError(f"image root is not a regular directory: {root}")
    return root


def _manifest_image_paths(manifest: Mapping[str, Any], image_root: Path) -> list[tuple[str, Path]]:
    """Resolve and hash-check only images named by the frozen manifest."""

    result: list[tuple[str, Path]] = []
    for item in cast(list[Mapping[str, Any]], manifest["images"]):
        name = cast(str, item["name"])
        path = (image_root / name).resolve()
        try:
            path.relative_to(image_root)
        except ValueError as exc:
            raise BDDRunnerError("manifest image escapes image root") from exc
        if not path.is_file() or path.is_symlink():
            raise BDDRunnerError(f"manifest image is missing or not regular: {name}")
        digest, size = _sha256_file(path)
        if digest != item["sha256"] or size != item["bytes"]:
            raise BDDRunnerError(f"image bytes do not match frozen manifest: {name}")
        result.append((name, path))
    return result


def _assert_images_stable(manifest: Mapping[str, Any], image_root: Path) -> None:
    _manifest_image_paths(manifest, image_root)


def _runtime_lock(path: Path, *, packages: Sequence[str] | None = None) -> tuple[str, str]:
    """Write a deterministic dependency/runtime lock and return its hash/text."""

    package_names = tuple(
        packages
        if packages is not None
        else ("onnxruntime", "numpy", "Pillow", "pydantic", "roadsense-perception", "bdd100k")
    )
    lines = [
        f"python=={platform.python_version()}",
        f"platform=={platform.platform()}",
    ]
    for name in sorted(set(package_names), key=str.casefold):
        try:
            version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            version = "unavailable"
        lines.append(f"{name}=={version}")
    payload = "\n".join(lines) + "\n"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8", newline="\n")
    except OSError as exc:
        raise BDDRunnerError(f"cannot write runtime lock: {path}") from exc
    return _sha256_file(path)[0], payload


def _evaluator_runtime_lock(
    evaluator_python: Path,
    output_path: Path,
    *,
    require_formal_dependencies: bool = False,
) -> tuple[str, str, dict[str, str]]:
    """Capture dependency versions from the *isolated evaluator* interpreter."""

    # Metadata lookup is enough and cannot load model code.  Use a small
    # multiline ``-c`` script so missing optional packages are represented as
    # ``unavailable`` rather than making the evaluator lock incomplete.
    script = (
        "import importlib.metadata as m\n"
        "names=('bdd100k','scalabel','numpy','pydantic','motmetrics','pycocotools')\n"
        "out=[]\n"
        "for n in names:\n"
        "    try:\n"
        "        v=m.version(n)\n"
        "    except m.PackageNotFoundError:\n"
        "        v='unavailable'\n"
        "    out.append(f'{n}=={v}')\n"
        "print('\\n'.join(out))\n"
    )
    try:
        completed = subprocess.run(
            [str(evaluator_python), "-c", script],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="strict",
            check=False,
            timeout=60,
            env={**os.environ, "PYTHONNOUSERSITE": "1"},
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise EvaluatorError("unable to inspect evaluator dependency versions") from exc
    if completed.returncode != 0:
        raise EvaluatorError("evaluator dependency version probe failed")
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    packages: dict[str, str] = {}
    for line in lines:
        if "==" in line:
            name, version = line.split("==", 1)
            if (
                not name
                or not version
                or any(ord(char) < 32 or ord(char) == 127 for char in name + version)
                or any(char in name + version for char in ("/", "\\", ":", "@"))
                or len(name) > 128
                or len(version) > 128
            ):
                raise EvaluatorError("evaluator dependency probe returned an unsafe package value")
            packages[name] = version
    if not packages:
        raise EvaluatorError("evaluator dependency probe returned no packages")
    if require_formal_dependencies and packages != BDD100K_REQUIRED_EVALUATOR_PACKAGES:
        raise EvaluatorError(
            "pinned BDD100K evaluator packages do not match the validated runtime lock"
        )
    lock_text = "\n".join(f"{name}=={packages[name]}" for name in sorted(packages)) + "\n"
    lock_path = output_path / "evaluator-runtime-lock.txt"
    lock_path.write_text(lock_text, encoding="utf-8", newline="\n")
    return _sha256_file(lock_path)[0], lock_text, packages


def _validate_devkit_checkout(path: Path) -> None:
    """Bind a formal evaluator invocation to the pinned clean Git checkout."""

    git = shutil.which("git")
    if git is None:
        raise EvaluatorError("git is required to verify the formal BDD100K devkit checkout")

    def run_git(*arguments: str) -> str:
        try:
            completed = subprocess.run(
                [git, "-C", str(path), *arguments],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="strict",
                check=False,
                timeout=60,
                env={**os.environ, "GIT_CONFIG_NOSYSTEM": "1"},
            )
        except (OSError, UnicodeError, subprocess.TimeoutExpired) as exc:
            raise EvaluatorError("unable to inspect the formal BDD100K devkit checkout") from exc
        if completed.returncode != 0:
            raise EvaluatorError("evaluator-cwd is not a readable BDD100K Git checkout")
        return completed.stdout.strip()

    checkout_root = Path(run_git("rev-parse", "--show-toplevel")).resolve()
    if checkout_root != path:
        raise EvaluatorError("evaluator-cwd must be the BDD100K checkout root")
    if run_git("rev-parse", "--verify", "HEAD") != DEVKIT_COMMIT:
        raise EvaluatorError(f"BDD100K devkit checkout must be pinned to {DEVKIT_COMMIT}")
    if run_git("status", "--porcelain=v1", "--untracked-files=all"):
        raise EvaluatorError("formal BDD100K devkit checkout must have a clean worktree")
    entrypoint = path / "bdd100k" / "eval" / "run.py"
    if not entrypoint.is_file() or entrypoint.is_symlink():
        raise EvaluatorError("formal BDD100K devkit checkout is missing bdd100k/eval/run.py")


def _validate_evaluator_import_origin(evaluator_python: Path, checkout: Path) -> None:
    """Ensure the isolated interpreter imports bdd100k from the pinned checkout bytes."""

    script = (
        "from pathlib import Path\n"
        "import bdd100k.eval.run as module\n"
        "print(Path(module.__file__).resolve())\n"
    )
    try:
        completed = subprocess.run(
            [str(evaluator_python), "-c", script],
            cwd=str(checkout),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="strict",
            check=False,
            timeout=60,
            env={**os.environ, "PYTHONNOUSERSITE": "1", "PYTHONPATH": str(checkout)},
        )
    except (OSError, UnicodeError, subprocess.TimeoutExpired) as exc:
        raise EvaluatorError("unable to verify the BDD100K evaluator import origin") from exc
    if completed.returncode != 0:
        raise EvaluatorError("BDD100K evaluator import-origin probe failed")
    try:
        imported = Path(completed.stdout.strip()).resolve()
        imported.relative_to(checkout)
    except (OSError, ValueError) as exc:
        raise EvaluatorError("evaluator Python must import bdd100k from evaluator-cwd") from exc
    expected = (checkout / "bdd100k" / "eval" / "run.py").resolve()
    if imported != expected:
        raise EvaluatorError("evaluator Python imported an unexpected bdd100k entrypoint")


def _validate_onnx_session(session: Any) -> tuple[Any, Any]:
    return _coco_validate_onnx_session(session)


def _model_metadata(model_path: Path, model_sha256: str | None) -> dict[str, Any]:
    if not model_path.is_file() or model_path.is_symlink():
        raise BDDRunnerError("model must be a regular local file")
    if model_path.suffix.lower() != ".onnx":
        raise BDDRunnerError("YOLO11 model must be an .onnx file")
    digest, size = _sha256_file(model_path)
    if model_sha256 is not None and digest != _strict_sha256(model_sha256, field="model_sha256"):
        raise BDDRunnerError("model SHA-256 does not match --model-sha256")
    return {"path_name": model_path.name, "sha256": digest, "bytes": size}


def _write_model_manifest(
    output_path: Path, model_meta: Mapping[str, Any]
) -> tuple[str, dict[str, Any]]:
    """Write a portable model provenance manifest and return its digest."""

    manifest: dict[str, Any] = {
        "schema_version": "roadsense.bdd100k-detection-model/v1",
        "model_id": "ultralytics-yolo11n-coco-onnx",
        "artifact_sha256": model_meta["sha256"],
        "artifact_bytes": model_meta["bytes"],
        "artifact_format": "onnx",
        "framework": "ultralytics",
        "framework_version": "8.3.237",
        "runtime": "onnxruntime",
        "input": {"size": [INPUT_SIZE, INPUT_SIZE], "layout": "NCHW", "color_order": "RGB"},
        "tasks": ["detection"],
        "source": MODEL_SOURCE_URL,
        "license_id": "AGPL-3.0",
        "ontology": {
            "source": "COCO-80",
            "target": list(BDD100K_DETECTION_CATEGORIES),
            "mapping": dict(sorted(COCO_TO_BDD.items())),
        },
        "claim_boundary": (
            "COCO-pretrained cross-domain baseline; no BDD training data or weights are claimed."
        ),
    }
    manifest_sha = canonical_sha256(manifest)
    write_json_atomic(output_path / "model-manifest.json", manifest)
    return manifest_sha, manifest


def _session_runner(
    session: Any,
) -> tuple[str, str, Any, Any]:
    input_meta, output_meta = _validate_onnx_session(session)
    return input_meta.name, output_meta.name, input_meta, output_meta


def run_inference(
    *,
    model: Path | str,
    image_manifest: Path | str | Mapping[str, Any],
    dataset_root: Path | str | None = None,
    images_root: Path | str | None = None,
    output_dir: Path | str,
    score_threshold: float = 0.25,
    nms_iou: float = 0.7,
    max_detections: int = 300,
    model_sha256: str | None = None,
    session: Any | None = None,
    session_factory: Callable[[str], Any] | None = None,
) -> dict[str, Any]:
    """Run YOLO11 inference from a frozen image manifest only.

    ``session``/``session_factory`` are dependency-injection seams for pure
    tests.  In production the function creates an ONNX Runtime CPU session.
    No argument or code path accepts a ground-truth label file.
    """

    if isinstance(score_threshold, bool) or not isinstance(score_threshold, (int, float)):
        raise BDDRunnerError("score_threshold must be numeric")
    if not math.isfinite(float(score_threshold)) or not 0.0 <= float(score_threshold) <= 1.0:
        raise BDDRunnerError("score_threshold must be in [0, 1]")
    if isinstance(nms_iou, bool) or not isinstance(nms_iou, (int, float)):
        raise BDDRunnerError("nms_iou must be numeric")
    if not math.isfinite(float(nms_iou)) or not 0.0 < float(nms_iou) <= 1.0:
        raise BDDRunnerError("nms_iou must be in (0, 1]")
    if (
        isinstance(max_detections, bool)
        or not isinstance(max_detections, int)
        or not 1 <= max_detections <= 10_000
    ):
        raise BDDRunnerError("max_detections must be in [1, 10000]")

    model_path = Path(model).expanduser().resolve()
    output_path = Path(output_dir).expanduser().resolve()
    manifest = (
        _validate_image_manifest(image_manifest, require_frozen=True)
        if isinstance(image_manifest, Mapping)
        else load_frozen_image_manifest(image_manifest)
    )
    root = _resolve_image_root(
        Path(dataset_root).expanduser().resolve() if dataset_root is not None else Path.cwd(),
        Path(images_root).expanduser().resolve() if images_root is not None else None,
    )
    image_paths = _manifest_image_paths(manifest, root)
    if output_path == root or root in output_path.parents:
        raise BDDRunnerError("output directory must be outside the image root")
    output_path.mkdir(parents=True, exist_ok=True)
    model_meta = _model_metadata(model_path, model_sha256)
    model_manifest_sha, _model_manifest = _write_model_manifest(output_path, model_meta)
    lock_hash, _lock_text = _runtime_lock(output_path / "runtime-lock.txt")

    ort_session = session
    staging: Any | None = None
    if ort_session is None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise BDDRunnerError(
                "onnxruntime is required for infer; install the vision extra"
            ) from exc
        if session_factory is not None:
            ort_session = session_factory(str(model_path))
        else:
            ort_session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    del staging  # A reserved seam for callers that stage Unicode paths.
    input_name, output_name, input_meta, output_meta = _session_runner(ort_session)
    # Close the model TOCTOU window: ONNX Runtime may lazily read graph data
    # while the first request is running, so verify the immutable artifact
    # again immediately after graph validation and before image inference.
    post_load_model_sha, post_load_model_bytes = _sha256_file(model_path)
    if post_load_model_sha != model_meta["sha256"] or post_load_model_bytes != model_meta["bytes"]:
        raise BDDRunnerError("model artifact changed while the ONNX session was initialized")

    detections_by_image: dict[str, Sequence[Any]] = {}
    per_image: list[dict[str, Any]] = []
    inference_ms: list[float] = []
    dropped_total: Counter[str] = Counter()
    total_raw = 0
    total_mapped = 0
    for image_name, image_path in image_paths:
        tensor, scale, pad_x, pad_y, width, height = _letterbox(image_path)
        started = time.perf_counter_ns()
        output = ort_session.run([output_name], {input_name: tensor})[0]
        elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000.0
        inference_ms.append(elapsed_ms)
        decoded = _decode_predictions(
            np.asarray(output),
            scale=scale,
            pad_x=pad_x,
            pad_y=pad_y,
            width=width,
            height=height,
            score_threshold=float(score_threshold),
            nms_iou=float(nms_iou),
            max_detections=max_detections,
        )
        labels, dropped = detections_to_bdd_labels(decoded)
        # Keep the original detections for the pure document builder; this
        # avoids converting a box twice and lets tests inspect both seams.
        detections_by_image[image_name] = decoded
        dropped_total.update(dropped)
        total_raw += len(decoded)
        total_mapped += len(labels)
        per_image.append(
            {
                "name": image_name,
                "width": width,
                "height": height,
                "raw_detection_count": len(decoded),
                "mapped_detection_count": len(labels),
                "dropped_coco_detections": dropped,
                "inference_wall_time_ms": elapsed_ms,
            }
        )

    # Re-hash after inference to close the image TOCTOU window.  Labels are not
    # involved in either check.
    _assert_images_stable(manifest, root)
    frames, mapping_metadata = build_prediction_document(
        [cast(str, item["name"]) for item in manifest["images"]], detections_by_image
    )
    prediction_document: list[dict[str, Any]] = frames
    prediction_path = output_path / "predictions.json"
    write_json_atomic(prediction_path, prediction_document)
    prediction_sha, prediction_bytes = _sha256_file(prediction_path)
    prediction_canonical_sha = canonical_sha256(prediction_document)
    inference_config: dict[str, Any] = {
        "schema_version": "roadsense.bdd100k-detection-inference-config/v1",
        "dataset": {"task": "detection", "release": "2020", "split": "val"},
        "image_manifest_sha256": manifest["manifest_sha256"],
        "model_manifest_sha256": model_manifest_sha,
        "preprocessing": {
            "resize": "letterbox",
            "size": [INPUT_SIZE, INPUT_SIZE],
            "padding": 114,
            "scale": 1.0 / 255.0,
            "color_order": "RGB",
        },
        "postprocessing": {
            "score_threshold": float(score_threshold),
            "nms_iou": float(nms_iou),
            "max_detections": max_detections,
        },
        "ontology_map_sha256": canonical_sha256(COCO_TO_BDD),
    }
    write_json_atomic(output_path / "inference-config.json", inference_config)
    inference_config_sha, _ = _sha256_file(output_path / "inference-config.json")
    metadata: dict[str, Any] = {
        "schema_version": INFERENCE_RECEIPT_SCHEMA,
        "stage": "infer",
        "dataset": {
            "dataset_id": "BDD100K",
            "task": "detection",
            "release": "2020",
            "split": "val",
            "image_manifest_sha256": cast(str, manifest["manifest_sha256"]),
            "images_tree_sha256": cast(str, manifest["images_tree_sha256"]),
            "image_count": len(image_paths),
        },
        "model": {
            **model_meta,
            "input_size": [INPUT_SIZE, INPUT_SIZE],
            "manifest_sha256": model_manifest_sha,
        },
        "runtime": {
            "python": platform.python_version(),
            "onnxruntime": _package_version("onnxruntime"),
            "numpy": _package_version("numpy"),
            "providers": list(ort_session.get_providers()),
            "input_name": input_name,
            "input_shape": list(input_meta.shape),
            "output_name": output_name,
            "output_shape": list(output_meta.shape),
            "runtime_lock_sha256": lock_hash,
        },
        "inference": {
            "config_sha256": inference_config_sha,
            "score_threshold": float(score_threshold),
            "nms_iou": float(nms_iou),
            "max_detections": max_detections,
            "total_raw_detections": total_raw,
            "total_mapped_detections": total_mapped,
            "per_image": per_image,
            "total_wall_time_ms": float(sum(inference_ms)),
            "mean_wall_time_ms": float(np.mean(inference_ms)) if inference_ms else 0.0,
        },
        "ontology": {
            **mapping_metadata,
            "mapping_sha256": canonical_sha256(COCO_TO_BDD),
            "dropped_coco_detections": dict(sorted(dropped_total.items())),
        },
        "prediction": {
            "path": prediction_path.name,
            "sha256": prediction_sha,
            "canonical_sha256": prediction_canonical_sha,
            "bytes": prediction_bytes,
            "schema_version": PREDICTION_SCHEMA,
        },
        "claim_boundary": (
            "COCO YOLO11n weights adapted to eight representable BDD categories; "
            "rider and traffic sign are intentionally not fabricated. Official metrics "
            "are available only after the separate bdd100k-devkit evaluate stage."
        ),
    }
    metadata["run_id"] = canonical_sha256(metadata)[:16]
    write_json_atomic(output_path / "inference-receipt.json", metadata)
    return metadata


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def _validate_prediction_document(
    value: object, *, expected_names: Sequence[str] | None = None
) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise EvaluatorError("BDD predictions must be a Scalabel JSON array")
    frames: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, frame in enumerate(value):
        if not isinstance(frame, Mapping):
            raise EvaluatorError(f"prediction frame {index} must be an object")
        name = frame.get("name")
        if not _safe_image_name(name):
            raise EvaluatorError(f"prediction frame {index} has an unsafe image name")
        assert isinstance(name, str)
        if name in seen:
            raise EvaluatorError("prediction document contains duplicate image names")
        seen.add(name)
        labels = frame.get("labels")
        if not isinstance(labels, list):
            raise EvaluatorError(f"prediction frame {name} has no labels array")
        for label_index, label in enumerate(labels):
            if not isinstance(label, Mapping):
                raise EvaluatorError(f"prediction label {name}:{label_index} must be an object")
            category = label.get("category")
            if category not in BDD100K_DETECTION_CATEGORIES:
                raise EvaluatorError(f"prediction label {name}:{label_index} has unknown category")
            score = label.get("score")
            if (
                isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not math.isfinite(float(score))
                or not 0.0 <= float(score) <= 1.0
            ):
                raise EvaluatorError(f"prediction label {name}:{label_index} has invalid score")
            box = label.get("box2d")
            if not isinstance(box, Mapping):
                raise EvaluatorError(f"prediction label {name}:{label_index} has no box2d")
            try:
                coordinates = tuple(float(box[key]) for key in ("x1", "y1", "x2", "y2"))
            except (KeyError, TypeError, ValueError) as exc:
                raise EvaluatorError(
                    f"prediction label {name}:{label_index} has invalid box2d"
                ) from exc
            if (
                not all(math.isfinite(item) for item in coordinates)
                or coordinates[2] <= coordinates[0]
                or coordinates[3] <= coordinates[1]
            ):
                raise EvaluatorError(f"prediction label {name}:{label_index} has invalid box2d")
        frames.append(dict(frame))
    if expected_names is not None:
        expected = list(expected_names)
        observed = [cast(str, frame["name"]) for frame in frames]
        if observed != expected:
            raise EvaluatorError(
                "prediction frames must exactly and canonically cover image manifest"
            )
    return frames


_OFFICIAL_METRIC_KEYS = (
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


def _parse_json_with_nan(raw: bytes) -> object:
    """Parse the devkit's JSON while retaining undefined metrics as NaN.

    The official evaluator serializes undefined class/size buckets as the
    non-standard ``NaN`` token.  We accept that token only at this isolated
    parser boundary and remove non-finite values before building a public
    receipt; no NaN is ever emitted by RoadSense evidence JSON.
    """

    def parse_constant(token: str) -> float:
        if token == "NaN":
            return float("nan")
        raise ValueError(f"unsupported non-finite JSON constant: {token}")

    return json.loads(raw.decode("utf-8", errors="strict"), parse_constant=parse_constant)


def _extract_official_metrics(value: object) -> dict[str, float]:
    """Extract finite official aggregate metrics from a devkit result object."""

    if not isinstance(value, Mapping):
        return {}
    metrics: dict[str, float] = {}
    for key in _OFFICIAL_METRIC_KEYS:
        raw_metric = value.get(key)
        candidate: object = None
        if isinstance(raw_metric, Sequence) and not isinstance(raw_metric, (str, bytes)):
            # The devkit stores per-class values first and an aggregate row
            # containing ``OVERALL`` second.
            for row in reversed(raw_metric):
                if isinstance(row, Mapping) and "OVERALL" in row:
                    candidate = row["OVERALL"]
                    break
        elif isinstance(raw_metric, Mapping):
            candidate = raw_metric.get("OVERALL")
        elif raw_metric is not None:
            candidate = raw_metric
        if isinstance(candidate, (int, float)) and not isinstance(candidate, bool):
            number = float(candidate)
            if math.isfinite(number):
                metrics[key] = number
    # A small fake evaluator used by offline contract tests may expose a
    # scalar mAP.  Preserve it as a finite diagnostic while official runs use
    # the AP/AP50/... keys above.
    for key, candidate in value.items():
        if key in metrics or key in _OFFICIAL_METRIC_KEYS or not isinstance(key, str):
            continue
        if isinstance(candidate, (int, float)) and not isinstance(candidate, bool):
            number = float(candidate)
            if math.isfinite(number):
                metrics[key] = number
    return dict(sorted(metrics.items()))


def _run_evaluator_subprocess(
    *,
    evaluator_python: Path | str,
    ground_truth: Path,
    predictions: Path,
    result_path: Path,
    cwd: Path | None,
    timeout_seconds: float,
) -> dict[str, Any]:
    """Invoke the official evaluator out of process and capture all streams."""

    if timeout_seconds <= 0 or not math.isfinite(timeout_seconds):
        raise EvaluatorError("timeout_seconds must be finite and positive")
    python_text = str(evaluator_python)
    python_executable = Path(evaluator_python).expanduser()
    if python_executable.exists():
        resolved_python = str(python_executable)
    else:
        resolved_python = shutil.which(python_text) or ""
    if not resolved_python:
        raise EvaluatorError(f"evaluator Python does not exist on PATH: {evaluator_python}")
    command = [
        resolved_python,
        "-m",
        DEVKIT_MODULE,
        "-t",
        "det",
        "-g",
        str(ground_truth),
        "-r",
        str(predictions),
        "--out-file",
        str(result_path),
    ]
    started = time.perf_counter_ns()
    try:
        completed = subprocess.run(
            command,
            cwd=None if cwd is None else str(cwd),
            capture_output=True,
            text=False,
            check=False,
            timeout=timeout_seconds,
            env={**os.environ, "PYTHONNOUSERSITE": "1", "PYTHONHASHSEED": "0"},
        )
    except subprocess.TimeoutExpired as exc:
        stdout = bytes(exc.stdout or b"")
        stderr = bytes(exc.stderr or b"")
        return {
            "command": command,
            "returncode": None,
            "timed_out": True,
            "stdout": stdout,
            "stderr": stderr,
            "elapsed_ms": (time.perf_counter_ns() - started) / 1_000_000.0,
        }
    return {
        "command": command,
        "returncode": int(completed.returncode),
        "timed_out": False,
        "stdout": bytes(completed.stdout),
        "stderr": bytes(completed.stderr),
        "elapsed_ms": (time.perf_counter_ns() - started) / 1_000_000.0,
    }


def _public_evaluator_command(command: Sequence[str]) -> list[str]:
    """Return a portable, path-redacted evaluator command for receipts.

    The subprocess needs absolute paths so that the official devkit can open
    the local GT, prediction, and result files.  Those paths are machine-local
    implementation details, however, and must never be copied into a public
    receipt.  Keep the command shape explicit rather than applying a generic
    string replacement: this makes it impossible for a future workspace path
    to leak if it happens to contain an unusual separator or filename.
    """

    if len(command) != 11:
        raise EvaluatorError("evaluator command has an unexpected shape")
    # Normalize both slash conventions so receipts remain safe even when a
    # Windows command is inspected on a POSIX host (or vice versa).
    executable = str(command[0]).replace("\\", "/").rsplit("/", 1)[-1]
    if not executable:
        raise EvaluatorError("evaluator command has no executable name")
    expected_prefix = ["-m", DEVKIT_MODULE, "-t", "det", "-g"]
    if list(command[1:6]) != expected_prefix:
        raise EvaluatorError("evaluator command has an unexpected module or task")
    if command[7] != "-r" or command[9] != "--out-file":
        raise EvaluatorError("evaluator command has unexpected path flags")
    return [
        executable,
        "-m",
        DEVKIT_MODULE,
        "-t",
        "det",
        "-g",
        "<GT>",
        "-r",
        "<PRED>",
        "--out-file",
        "<RESULT>",
    ]


def run_evaluation(
    *,
    ground_truth: Path | str,
    predictions: Path | str,
    output_dir: Path | str,
    evaluator_python: Path | str = sys.executable,
    evaluator_cwd: Path | str | None = None,
    image_manifest: Path | str | Mapping[str, Any] | None = None,
    timeout_seconds: float = 3600.0,
    role: str | None = None,
) -> dict[str, Any]:
    """Run the pinned BDD100K official detection evaluator independently."""

    gt_path = Path(ground_truth).expanduser().resolve()
    prediction_path = Path(predictions).expanduser().resolve()
    output_path = Path(output_dir).expanduser().resolve()
    if not gt_path.is_file() or gt_path.is_symlink():
        raise EvaluatorError("ground-truth JSON must be a regular local file")
    if not prediction_path.is_file() or prediction_path.is_symlink():
        raise EvaluatorError("prediction JSON must be a regular local file")
    if gt_path == prediction_path:
        raise EvaluatorError("ground truth and predictions must be separate files")
    prediction_payload = load_strict_json(prediction_path)
    expected_names: Sequence[str] | None = None
    manifest_digest: str | None = None
    if image_manifest is not None:
        manifest = (
            _validate_image_manifest(image_manifest, require_frozen=True)
            if isinstance(image_manifest, Mapping)
            else load_frozen_image_manifest(image_manifest)
        )
        expected_names = [cast(str, item["name"]) for item in manifest["images"]]
        manifest_digest = cast(str, manifest["manifest_sha256"])
    if role is not None and role not in {"independent_a", "independent_b"}:
        raise EvaluatorError("evaluation role must be independent_a or independent_b")
    if expected_names is not None and len(expected_names) == 10_000 and role is None:
        raise EvaluatorError(
            "formal BDD100K evaluation requires --role independent_a or independent_b"
        )
    frames = _validate_prediction_document(prediction_payload, expected_names=expected_names)
    output_path.mkdir(parents=True, exist_ok=True)
    gt_sha, gt_bytes = _sha256_file(gt_path)
    prediction_sha, prediction_bytes = _sha256_file(prediction_path)
    cwd_path = None if evaluator_cwd is None else Path(evaluator_cwd).expanduser().resolve()
    if cwd_path is not None and not cwd_path.is_dir():
        raise EvaluatorError("evaluator-cwd must be a directory")
    formal_lane = expected_names is not None and len(expected_names) == 10_000
    if formal_lane:
        if cwd_path is None:
            raise EvaluatorError("formal BDD100K evaluation requires --evaluator-cwd")
        _validate_devkit_checkout(cwd_path)
    evaluator_python_path = Path(evaluator_python).expanduser()
    if formal_lane:
        assert cwd_path is not None
        _validate_evaluator_import_origin(evaluator_python_path, cwd_path)
    evaluator_config = {
        "schema_version": "roadsense.bdd100k-detection-evaluator-config/v1",
        "evaluator_id": DEVKIT_ID,
        "module": DEVKIT_MODULE,
        "task": "det",
        "dataset": {"dataset": "BDD100K", "release": "2020", "split": "val"},
        "devkit_commit": DEVKIT_COMMIT,
        "iou_threshold": 0.5,
        "ignore_iof_threshold": 0.5,
    }
    write_json_atomic(output_path / "evaluator-config.json", evaluator_config)
    evaluator_config_sha, _ = _sha256_file(output_path / "evaluator-config.json")
    evaluator_lock_sha, _evaluator_lock_text, evaluator_packages = _evaluator_runtime_lock(
        evaluator_python_path,
        output_path,
        # Only the official 10k lane is subject to the pycocotools pin.  Tiny
        # evaluator fixtures remain useful for plumbing tests and are marked
        # non-benchmark by their reduced image manifest.
        require_formal_dependencies=formal_lane,
    )
    result_target = output_path / "evaluator-result.json"
    if result_target == gt_path or result_target == prediction_path:
        raise EvaluatorError("evaluator output must not overwrite ground truth or predictions")
    # Do not let a previous invocation's result masquerade as fresh evaluator
    # evidence when a wrapper exits successfully but forgets to write its
    # ``--out-file``.  The output directory is the current run's workspace, so
    # replacing this well-known generated artifact is intentional.
    if result_target.is_symlink():
        raise EvaluatorError("evaluator result path must not be a symlink")
    if result_target.exists():
        if not result_target.is_file():
            raise EvaluatorError("evaluator result path must be a regular file")
        try:
            result_target.unlink()
        except OSError as exc:
            raise EvaluatorError("unable to clear stale evaluator result") from exc
    result = _run_evaluator_subprocess(
        evaluator_python=evaluator_python_path,
        ground_truth=gt_path,
        predictions=prediction_path,
        result_path=result_target,
        cwd=cwd_path,
        timeout_seconds=float(timeout_seconds),
    )
    stdout = cast(bytes, result["stdout"])
    stderr = cast(bytes, result["stderr"])
    stdout_sha = _sha256_bytes(stdout)
    stderr_sha = _sha256_bytes(stderr)
    combined_sha = _sha256_bytes(stdout + b"\x00" + stderr)
    _write_generated_bytes(output_path / "evaluator.stdout.txt", stdout, field="evaluator stdout")
    _write_generated_bytes(output_path / "evaluator.stderr.txt", stderr, field="evaluator stderr")
    result_file = output_path / "evaluator-result.json"
    result_artifact = result_file
    result_source = "file"
    result_sha: str | None = None
    result_bytes = 0
    parsed_result: object = None
    fallback_json_line: bytes | None = None
    if result_file.is_file() and not result_file.is_symlink():
        result_sha, result_bytes = _sha256_file(result_file)
        try:
            parsed_result = _parse_json_with_nan(result_file.read_bytes())
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
            raise EvaluatorError("official evaluator result is not valid JSON") from exc
    if parsed_result is None:
        # Contract-test evaluators and older devkit wrappers may omit
        # ``--out-file`` output; retain a JSON object printed on stdout as a
        # fallback, while production receipts prefer the result file.
        try:
            full_stdout_candidate = _parse_json_with_nan(stdout)
        except (UnicodeError, ValueError, json.JSONDecodeError):
            full_stdout_candidate = None
        if isinstance(full_stdout_candidate, (Mapping, list)):
            parsed_result = full_stdout_candidate
            fallback_json_line = stdout
        for line in reversed(stdout.decode("utf-8", errors="replace").splitlines()):
            if parsed_result is not None:
                break
            text = line.strip()
            if not text:
                continue
            try:
                candidate = _parse_json_with_nan(text.encode("utf-8"))
            except (UnicodeError, ValueError, json.JSONDecodeError):
                continue
            if isinstance(candidate, (Mapping, list)):
                parsed_result = candidate
                # Keep the exact JSON line (including non-standard NaN values
                # emitted by the devkit) as the fallback artifact.  It is
                # hashed as-is and never loaded through the strict public JSON
                # reader, preserving provenance without leaking local paths.
                fallback_json_line = text.encode("utf-8") + b"\n"
                break
    if result_sha is None:
        # A few wrappers honor the evaluator command but omit ``--out-file``.
        # Materialize a sibling artifact instead of pointing the receipt at a
        # file that does not exist.  If a JSON line was found, retain that
        # exact line; otherwise retain stdout as a diagnostic artifact and
        # mark the evaluation failed below because no result was available.
        result_source = "stdout_fallback"
        result_artifact = output_path / "evaluator-result.stdout-fallback.json"
        fallback_payload = fallback_json_line if fallback_json_line is not None else stdout
        _write_generated_bytes(result_artifact, fallback_payload, field="evaluator fallback result")
        result_sha, result_bytes = _sha256_file(result_artifact)
    returncode = result["returncode"]
    # Exit code alone is not sufficient evidence: a successful wrapper that
    # emitted neither a result file nor parseable JSON must not be promoted to
    # a benchmark claim.
    status = (
        "ok"
        if returncode == 0 and not result["timed_out"] and parsed_result is not None
        else "failed"
    )
    official_metrics = _extract_official_metrics(parsed_result)
    receipt: dict[str, Any] = {
        "schema_version": EVALUATION_RECEIPT_SCHEMA,
        "stage": "evaluate",
        "status": status,
        "role": role,
        "dataset": {
            "ground_truth_sha256": gt_sha,
            "ground_truth_bytes": gt_bytes,
            "split": "val",
            "image_manifest_sha256": manifest_digest,
        },
        "prediction": {
            "sha256": prediction_sha,
            "bytes": prediction_bytes,
            "frame_count": len(frames),
        },
        "evaluator": {
            "id": DEVKIT_ID,
            "module": DEVKIT_MODULE,
            "commit": DEVKIT_COMMIT,
            "python": str(evaluator_python).replace("\\", "/").rsplit("/", 1)[-1],
            "command": _public_evaluator_command(cast(Sequence[str], result["command"])),
            "returncode": returncode,
            "timed_out": result["timed_out"],
            "elapsed_ms": result["elapsed_ms"],
            "runtime_lock_sha256": evaluator_lock_sha,
            "evaluator_config_sha256": evaluator_config_sha,
            "packages": evaluator_packages,
            "stdout_sha256": stdout_sha,
            "stderr_sha256": stderr_sha,
            "streams_sha256": combined_sha,
            "result_sha256": result_sha,
            "result_bytes": result_bytes,
            "result_source": result_source,
            "metrics": official_metrics,
            # The raw devkit result is retained as a local file and bound by
            # this digest.  Its undefined buckets may contain NaN, so the
            # receipt stores only a portable file reference/hash here.
            "result": {
                "path": result_artifact.name,
                "sha256": result_sha,
                "bytes": result_bytes,
                "source": result_source,
            },
        },
        "claim_boundary": (
            "Metrics are official BDD100K Detection 2020 val metrics only when this "
            "receipt has status=ok and the pinned bdd100k-devkit command completed."
        ),
    }
    receipt["run_id"] = canonical_sha256(receipt)[:16]
    write_json_atomic(output_path / "evaluation-receipt.json", receipt)
    if status != "ok":
        raise EvaluatorError(
            f"BDD100K devkit failed (returncode={returncode}, timed_out={result['timed_out']})"
        )
    return receipt


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    freeze = subparsers.add_parser("freeze", help="bind an image manifest for inference")
    freeze.add_argument("--manifest", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)

    infer = subparsers.add_parser("infer", help="run local YOLO11 ONNX inference")
    infer.add_argument("--model", type=Path, required=True)
    infer.add_argument("--image-manifest", type=Path, required=True)
    infer.add_argument("--dataset-root", type=Path, default=Path("."))
    infer.add_argument("--images-root", type=Path)
    infer.add_argument("--output-dir", type=Path, required=True)
    infer.add_argument("--model-sha256")
    infer.add_argument("--score-threshold", type=float, default=0.25)
    infer.add_argument("--nms-iou", type=float, default=0.7)
    infer.add_argument("--max-detections", type=int, default=300)

    evaluate = subparsers.add_parser("evaluate", help="invoke isolated BDD100K devkit")
    evaluate.add_argument("--ground-truth", "--gt", type=Path, required=True)
    evaluate.add_argument("--predictions", "--pred", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    evaluate.add_argument("--image-manifest", type=Path)
    evaluate.add_argument("--evaluator-python", type=Path, default=Path(sys.executable))
    evaluate.add_argument("--evaluator-cwd", type=Path)
    evaluate.add_argument("--timeout-seconds", type=float, default=3600.0)
    evaluate.add_argument(
        "--role",
        choices=("independent_a", "independent_b"),
        help="Required independent evaluator role for the formal 10,000-image lane.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "freeze":
            result = freeze_image_manifest(args.manifest, args.output)
            print(
                json.dumps(
                    {"status": "ok", "manifest_sha256": result["manifest_sha256"]}, sort_keys=True
                )
            )
        elif args.command == "infer":
            result = run_inference(
                model=args.model,
                image_manifest=args.image_manifest,
                dataset_root=args.dataset_root,
                images_root=args.images_root,
                output_dir=args.output_dir,
                model_sha256=args.model_sha256,
                score_threshold=args.score_threshold,
                nms_iou=args.nms_iou,
                max_detections=args.max_detections,
            )
            print(
                json.dumps(
                    {
                        "status": "ok",
                        "run_id": result["run_id"],
                        "prediction": result["prediction"],
                    },
                    sort_keys=True,
                )
            )
        else:
            result = run_evaluation(
                ground_truth=args.ground_truth,
                predictions=args.predictions,
                output_dir=args.output_dir,
                evaluator_python=args.evaluator_python,
                evaluator_cwd=args.evaluator_cwd,
                image_manifest=args.image_manifest,
                timeout_seconds=args.timeout_seconds,
                role=args.role,
            )
            print(
                json.dumps(
                    {"status": "ok", "run_id": result["run_id"], "evaluator": result["evaluator"]},
                    sort_keys=True,
                )
            )
        return 0
    except (
        BDDRunnerError,
        ImportError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        MemoryError,
    ) as exc:
        print(f"BDD100K benchmark stage failed: {exc}", file=sys.stderr)
        return 2


__all__ = [
    "BDD100K_CATEGORIES",
    "BDD100K_DETECTION_CATEGORIES",
    "COCO_CLASS_NAMES",
    "COCO_TO_BDD",
    "COCO_TO_BDD_CATEGORY",
    "DEVKIT_COMMIT",
    "DEVKIT_ID",
    "DEVKIT_MODULE",
    "EVALUATION_RECEIPT_SCHEMA",
    "FROZEN_IMAGE_MANIFEST_SCHEMA",
    "IMAGE_MANIFEST_SCHEMA",
    "INFERENCE_RECEIPT_SCHEMA",
    "PREDICTION_SCHEMA",
    "REQUIRED_PYCOCOTOOLS_VERSION",
    "UNSUPPORTED_BDD_CATEGORIES",
    "BDDRunnerError",
    "FrozenManifestError",
    "_decode_predictions",
    "_decode_yolo11_predictions",
    "_letterbox",
    "_load_frozen_image_manifest",
    "_public_evaluator_command",
    "_runtime_lock",
    "_sha256_file",
    "_validate_devkit_checkout",
    "_validate_evaluator_import_origin",
    "build_prediction_document",
    "build_prediction_frame",
    "compute_image_manifest_sha256",
    "detections_to_bdd_labels",
    "freeze_image_manifest",
    "load_frozen_image_manifest",
    "main",
    "map_coco_category",
    "run_evaluation",
    "run_inference",
    "validate_frozen_image_manifest",
]


if __name__ == "__main__":
    raise SystemExit(main())
