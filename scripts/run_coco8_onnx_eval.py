"""Run a small, reproducible real-data ONNX detection evaluation.

This runner is intentionally separate from the public Pages fixture.  It uses
operator-downloaded COCO8 images/YOLO labels and a locally verified Ultralytics
YOLO11n ONNX artifact, converts both sides to RoadSense ``FrameRecord``
bundles, and invokes the normal fail-closed local evaluator.  The resulting
numbers are development evidence on eight images (or four images when the
default ``val`` split is selected); they are not official COCO benchmark
results.

The script has no network code.  Download and license review happen outside
the runner.  Optional vision dependencies (Pillow and ONNX Runtime) are
imported only inside ``main`` so the core package and Pages build remain
dependency-light; NumPy is already a core project dependency.
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
import sys
import tempfile
import time
from pathlib import Path
from typing import cast

import numpy as np

from roadsense.adapters import (
    MODEL_ARTIFACT_SCHEMA,
    ModelArtifactManifest,
    verify_artifact_manifest,
)
from roadsense.contracts import BoxXYXY, DatasetManifest, Detection
from roadsense.json_io import canonical_sha256, write_json_atomic
from roadsense.local_eval import evaluate_local

MODEL_SOURCE_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n.onnx"
MODEL_SHA256 = "634279b40c07c6391472c51ad45b81ebc48706a9a1fe72dd3396322acd0c053b"
DATASET_SOURCE_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco8.zip"
DATASET_ARCHIVE_SHA256 = "54c67fe9ef88313e021ec0e92b73c200167bb0a86633e8df8658d832cca828c9"
COCO_CLASS_COUNT = 80
INPUT_SIZE = 640
COCO_CLASS_NAMES = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
)
if len(COCO_CLASS_NAMES) != COCO_CLASS_COUNT:
    raise RuntimeError("COCO class ontology must contain exactly 80 names")


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _tree_inventory(root: Path) -> tuple[list[dict[str, object]], str]:
    files: list[dict[str, object]] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest, size = _sha256_file(path)
        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": digest,
                "bytes": size,
            }
        )
    if not files:
        raise ValueError(f"dataset directory is empty: {root}")
    return files, canonical_sha256(files)


def _runtime_lock(path: Path) -> tuple[str, str]:
    """Write a small, local runtime lock and return its hash/content."""

    names = ("onnxruntime", "numpy", "Pillow", "pydantic", "roadsense-perception")
    lines = [f"python=={platform.python_version()}"]
    for name in names:
        try:
            lines.append(f"{name}=={importlib.metadata.version(name)}")
        except importlib.metadata.PackageNotFoundError:
            lines.append(f"{name}==unavailable")
    payload = "\n".join(lines) + "\n"
    path.write_text(payload, encoding="utf-8")
    return _sha256_file(path)[0], payload


def _common_root(left: Path, right: Path) -> Path:
    """Find a filesystem root containing both artifact and receipt files."""

    try:
        return Path(os.path.commonpath((str(left), str(right))))
    except ValueError as exc:
        raise ValueError("model and output directory must be on the same filesystem drive") from exc


def _parse_image_size(path: Path) -> tuple[int, int]:
    try:
        from PIL import Image

        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            width, height = image.size
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot read image {path}: {exc}") from exc
    if width < 1 or height < 1 or width > 100_000 or height > 100_000:
        raise ValueError(f"image has unsafe dimensions: {path}")
    return int(width), int(height)


def _load_truth(path: Path, *, width: int, height: int) -> tuple[Detection, ...]:
    detections: list[Detection] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"cannot read label file {path}: {exc}") from exc
    for line_number, line in enumerate(lines, start=1):
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"label {path}:{line_number} must contain five fields")
        try:
            category_id = int(fields[0])
            x_center, y_center, box_width, box_height = (float(item) for item in fields[1:])
        except ValueError as exc:
            raise ValueError(f"label {path}:{line_number} contains a non-numeric value") from exc
        if not 0 <= category_id < COCO_CLASS_COUNT:
            raise ValueError(f"label {path}:{line_number} has class {category_id}")
        values = (x_center, y_center, box_width, box_height)
        if any(not math.isfinite(item) for item in values) or any(
            item < 0.0 or item > 1.0 for item in values
        ):
            raise ValueError(f"label {path}:{line_number} has out-of-range coordinates")
        x_min = max(0.0, (x_center - box_width / 2.0) * width)
        y_min = max(0.0, (y_center - box_height / 2.0) * height)
        x_max = min(float(width), (x_center + box_width / 2.0) * width)
        y_max = min(float(height), (y_center + box_height / 2.0) * height)
        if x_max <= x_min or y_max <= y_min:
            raise ValueError(f"label {path}:{line_number} has an empty box")
        detections.append(
            Detection(
                category_id=category_id,
                score=1.0,
                bbox=BoxXYXY(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max),
            )
        )
    return tuple(detections)


def _letterbox(path: Path) -> tuple[np.ndarray, float, int, int, int, int]:
    """Return NCHW float32 input and inverse letterbox parameters."""

    from PIL import Image

    with Image.open(path) as source:
        image = source.convert("RGB")
        width, height = image.size
        scale = min(INPUT_SIZE / width, INPUT_SIZE / height)
        resized_width = max(1, round(width * scale))
        resized_height = max(1, round(height * scale))
        resized = image.resize((resized_width, resized_height), Image.Resampling.BILINEAR)
    pad_x = (INPUT_SIZE - resized_width) // 2
    pad_y = (INPUT_SIZE - resized_height) // 2
    canvas = Image.new("RGB", (INPUT_SIZE, INPUT_SIZE), (114, 114, 114))
    canvas.paste(resized, (pad_x, pad_y))
    array = np.asarray(canvas, dtype=np.float32) / 255.0
    return (
        np.transpose(array, (2, 0, 1))[None, ...],
        float(scale),
        pad_x,
        pad_y,
        int(width),
        int(height),
    )


def _box_iou(box: np.ndarray, others: np.ndarray) -> np.ndarray:
    left = np.maximum(box[0], others[:, 0])
    top = np.maximum(box[1], others[:, 1])
    right = np.minimum(box[2], others[:, 2])
    bottom = np.minimum(box[3], others[:, 3])
    intersection = np.maximum(0.0, right - left) * np.maximum(0.0, bottom - top)
    area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    other_area = np.maximum(0.0, others[:, 2] - others[:, 0]) * np.maximum(
        0.0, others[:, 3] - others[:, 1]
    )
    return cast(np.ndarray, intersection / np.maximum(area + other_area - intersection, 1e-12))


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
) -> tuple[Detection, ...]:
    if output.ndim != 3 or output.shape[0] != 1:
        raise ValueError(f"unexpected ONNX output shape: {output.shape}")
    values = output[0]
    if values.shape[0] == 4 + COCO_CLASS_COUNT:
        values = values.T
    elif values.shape[1] != 4 + COCO_CLASS_COUNT:
        raise ValueError(f"expected 84-channel YOLO output, got {output.shape}")
    if values.shape[1] != 4 + COCO_CLASS_COUNT:
        raise ValueError(f"expected 84-channel YOLO output, got {output.shape}")
    class_scores = values[:, 4:]
    class_ids = np.argmax(class_scores, axis=1)
    scores = class_scores[np.arange(class_scores.shape[0]), class_ids]
    keep = np.flatnonzero(scores >= score_threshold)
    if keep.size == 0:
        return ()
    boxes = values[keep, :4].astype(np.float64, copy=False)
    selected_scores = scores[keep].astype(np.float64, copy=False)
    selected_classes = class_ids[keep].astype(np.int64, copy=False)
    xyxy = np.empty_like(boxes)
    xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
    xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
    xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
    xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
    xyxy[:, [0, 2]] = (xyxy[:, [0, 2]] - pad_x) / scale
    xyxy[:, [1, 3]] = (xyxy[:, [1, 3]] - pad_y) / scale
    xyxy[:, [0, 2]] = np.clip(xyxy[:, [0, 2]], 0.0, float(width))
    xyxy[:, [1, 3]] = np.clip(xyxy[:, [1, 3]], 0.0, float(height))

    selected: list[int] = []
    for category_id in sorted({int(item) for item in selected_classes}):
        candidates = np.flatnonzero(selected_classes == category_id)
        order = candidates[np.argsort(-selected_scores[candidates], kind="stable")]
        while order.size:
            current = int(order[0])
            selected.append(current)
            if order.size == 1:
                break
            overlaps = _box_iou(xyxy[current], xyxy[order[1:]])
            order = order[1:][overlaps <= nms_iou]
    selected.sort(
        key=lambda index: (-float(selected_scores[index]), int(selected_classes[index]), index)
    )
    result: list[Detection] = []
    for index in selected[:max_detections]:
        x_min, y_min, x_max, y_max = (float(item) for item in xyxy[index])
        if x_max <= x_min or y_max <= y_min:
            continue
        result.append(
            Detection(
                category_id=int(selected_classes[index]),
                score=float(selected_scores[index]),
                bbox=BoxXYXY(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max),
            )
        )
    return tuple(result)


def _image_paths(dataset_root: Path, split: str) -> list[Path]:
    image_dir = dataset_root / "images" / split
    label_dir = dataset_root / "labels" / split
    if not image_dir.is_dir() or not label_dir.is_dir():
        raise ValueError(f"COCO8 split directories are missing: {image_dir}, {label_dir}")
    paths = sorted(image_dir.glob("*.jpg"))
    if not paths:
        raise ValueError(f"no images found in {image_dir}")
    for image in paths:
        if not (label_dir / f"{image.stem}.txt").is_file():
            raise ValueError(f"missing label for image: {image}")
    return paths


def _relative_path(path: Path, base: Path) -> str:
    """Return a portable local path from ``base`` to ``path``."""

    try:
        relative = Path(os.path.relpath(path, base))
    except ValueError as exc:
        raise ValueError(
            "output-dir and model must be on the same filesystem drive so the "
            "evaluation spec can use a relative artifact root"
        ) from exc
    return relative.as_posix()


def _ort_model_reference(
    model_path: Path, expected_sha256: str
) -> tuple[str, tempfile.TemporaryDirectory[str] | None]:
    """Return an ORT-safe path, staging Unicode paths in an ASCII temp folder."""

    cwd = Path.cwd().resolve()
    try:
        relative = model_path.relative_to(cwd).as_posix()
    except ValueError:
        relative = ""
    if relative and relative.isascii():
        return relative, None
    absolute = str(model_path)
    if absolute.isascii():
        return absolute, None
    # Keep the fallback relative to the current checkout when possible.  The
    # workspace itself may have a non-ASCII parent directory on Windows, but
    # ORT receives the ASCII relative spelling and lets Windows resolve it.
    staging_parent = cwd / ".tmp"
    try:
        staging_parent.mkdir(parents=True, exist_ok=True)
        staging = tempfile.TemporaryDirectory(prefix="roadsense_ort_", dir=str(staging_parent))
        staged_path = Path(staging.name) / "model.onnx"
        ort_reference = staged_path.relative_to(cwd).as_posix()
        if not ort_reference.isascii():
            staging.cleanup()
            raise OSError("checkout-relative ORT staging path is not ASCII")
    except OSError:
        staging = tempfile.TemporaryDirectory(prefix="roadsense_ort_")
        staged_path = Path(staging.name) / "model.onnx"
        ort_reference = str(staged_path)
        if not ort_reference.isascii():
            staging.cleanup()
            raise ValueError(
                "unable to create an ASCII model path for this ONNX Runtime build; "
                "choose an ASCII model/output directory"
            )
    try:
        shutil.copyfile(model_path, staged_path)
        staged_hash, staged_size = _sha256_file(staged_path)
    except OSError:
        staging.cleanup()
        raise
    original_size = model_path.stat().st_size
    if staged_hash != expected_sha256 or staged_size != original_size:
        staging.cleanup()
        raise ValueError("temporary ASCII model copy failed hash/size verification")
    return ort_reference, staging


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--score-threshold", type=float, default=0.25)
    parser.add_argument("--nms-iou", type=float, default=0.7)
    parser.add_argument("--max-detections", type=int, default=300)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if not 0.0 <= args.score_threshold <= 1.0:
            raise ValueError("--score-threshold must be in [0, 1]")
        if not 0.0 < args.nms_iou <= 1.0:
            raise ValueError("--nms-iou must be in (0, 1]")
        if not 1 <= args.max_detections <= 10_000:
            raise ValueError("--max-detections must be in [1, 10000]")
        dataset_root = args.dataset_root.expanduser().resolve()
        model_path = args.model.expanduser().resolve()
        archive_path = args.archive.expanduser().resolve()
        output_dir = args.output_dir.expanduser().resolve()
        if not dataset_root.is_dir() or not model_path.is_file() or not archive_path.is_file():
            raise ValueError("dataset root, model, and archive must exist locally")
        archive_hash, archive_size = _sha256_file(archive_path)
        if archive_hash != DATASET_ARCHIVE_SHA256:
            raise ValueError(
                f"unexpected COCO8 archive SHA-256: expected {DATASET_ARCHIVE_SHA256}, got {archive_hash}"
            )
        model_hash, model_size = _sha256_file(model_path)
        if model_hash != MODEL_SHA256:
            raise ValueError(f"unexpected model SHA-256: expected {MODEL_SHA256}, got {model_hash}")
        images = _image_paths(dataset_root, args.split)
        _inventory, tree_hash = _tree_inventory(dataset_root)
        output_dir.mkdir(parents=True, exist_ok=True)
        artifact_root = _common_root(model_path.parent, output_dir)
        artifact_root_ref = _relative_path(artifact_root, output_dir)
        artifact_path_ref = _relative_path(model_path, artifact_root)
        lock_path = output_dir / "runtime-lock.txt"
        lock_hash, _ = _runtime_lock(lock_path)
        model_manifest = ModelArtifactManifest.model_validate(
            {
                "schema_version": MODEL_ARTIFACT_SCHEMA,
                "artifact_id": "ultralytics-yolo11n-onnx-ultralytics-8.3.237",
                "artifact_path": artifact_path_ref,
                "artifact_sha256": model_hash,
                "artifact_size_bytes": model_size,
                "artifact_format": "onnx",
                "framework": "ultralytics",
                "framework_version": "8.3.237",
                "backend": "onnxruntime",
                "backend_version": importlib.metadata.version("onnxruntime"),
                "tasks": ["detection"],
                "ontology": [
                    {"category_id": index, "label": label}
                    for index, label in enumerate(COCO_CLASS_NAMES)
                ],
                "input": {
                    "width": INPUT_SIZE,
                    "height": INPUT_SIZE,
                    "channels": 3,
                    "dtype": "float32",
                    "layout": "NCHW",
                    "color_order": "RGB",
                    "coordinate_space": "letterboxed_640",
                },
                "output": {
                    "output_schema": "ultralytics.yolo-detect-84/v1",
                    "coordinate_space": "letterboxed_640",
                    "score_semantics": "class_probability_after_sigmoid",
                    "output_shape": [1, 84, None],
                },
                "preprocessing": {
                    "resize_mode": "letterbox",
                    "scale": 1.0 / 255.0,
                    "mean": [0.0, 0.0, 0.0],
                    "std": [1.0, 1.0, 1.0],
                    "pad_value": 114.0 / 255.0,
                },
                "runtime": {
                    "device": "cpu",
                    "precision": "fp32",
                    "runtime_name": "onnxruntime",
                    "runtime_version": importlib.metadata.version("onnxruntime"),
                    "opset": 22,
                    "deterministic": True,
                },
                "license_id": "AGPL-3.0",
                "source": MODEL_SOURCE_URL,
                "dependency_lock_path": _relative_path(
                    output_dir / "runtime-lock.txt", artifact_root
                ),
                "dependency_lock_sha256": lock_hash,
                "graph_metadata": {
                    "input_name": "images",
                    "output_name": "output0",
                    "nms_in_graph": False,
                    "model_image_size": [INPUT_SIZE, INPUT_SIZE],
                    "resize_interpolation": "bilinear",
                    "padding_anchor": "top_left_integer_floor",
                },
                "notes": (
                    "Asset release path is v8.4.0; embedded exporter metadata reports "
                    "Ultralytics 8.3.237 and AGPL-3.0. The checkpoint is an operator-provided "
                    "local input: review the license before any use, do not redistribute "
                    "weights, and do not call this an official COCO benchmark."
                ),
            }
        )
        manifest_path = output_dir / "model-artifact-manifest.json"
        write_json_atomic(manifest_path, model_manifest.model_dump(mode="json"))
        receipt = verify_artifact_manifest(model_manifest, artifact_root=artifact_root)

        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "onnxruntime is required; use an isolated CPython environment with the vision extra"
            ) from exc
        ort_model_path, _staging = _ort_model_reference(model_path, model_hash)
        session = ort.InferenceSession(ort_model_path, providers=["CPUExecutionProvider"])
        input_meta = session.get_inputs()[0]
        output_meta = session.get_outputs()[0]
        input_name = input_meta.name
        output_name = output_meta.name
        truth_sequences: list[dict[str, object]] = []
        prediction_sequences: list[dict[str, object]] = []
        inference_ms: list[float] = []
        total_detections = 0
        label_dir = dataset_root / "labels" / args.split
        for image_path in images:
            width, height = _parse_image_size(image_path)
            truth = _load_truth(label_dir / f"{image_path.stem}.txt", width=width, height=height)
            input_array, scale, pad_x, pad_y, _, _ = _letterbox(image_path)
            started = time.perf_counter_ns()
            output = session.run([output_name], {input_name: input_array})[0]
            inference_ms.append((time.perf_counter_ns() - started) / 1_000_000.0)
            prediction = _decode_predictions(
                np.asarray(output),
                scale=scale,
                pad_x=pad_x,
                pad_y=pad_y,
                width=width,
                height=height,
                score_threshold=args.score_threshold,
                nms_iou=args.nms_iou,
                max_detections=args.max_detections,
            )
            total_detections += len(prediction)
            sequence_id = f"coco8-{args.split}-{image_path.stem}"
            frame_base = {
                "frame_index": 0,
                "timestamp_ms": 0,
                "image_size": {"width": width, "height": height},
            }
            truth_sequences.append(
                {
                    "sequence_id": sequence_id,
                    "frames": [
                        {
                            **frame_base,
                            "detections": [item.model_dump(mode="json") for item in truth],
                        }
                    ],
                }
            )
            prediction_sequences.append(
                {
                    "sequence_id": sequence_id,
                    "frames": [
                        {
                            **frame_base,
                            "detections": [item.model_dump(mode="json") for item in prediction],
                        }
                    ],
                }
            )
        sequence_ids = [item["sequence_id"] for item in truth_sequences]
        dataset_manifest = DatasetManifest.model_validate(
            {
                "schema_version": "roadsense.dataset-manifest/v1",
                "dataset_name": "Ultralytics COCO8 development subset",
                "source_url": DATASET_SOURCE_URL,
                "license_id": "COCO-image-annotation-terms-plus-Ultralytics-AGPL-3.0",
                "tasks": ["detection"],
                "splits": {args.split: f"{len(sequence_ids)} complete image sequences"},
                "content_sha256": archive_hash,
                "evaluation_authorized": False,
                "frozen": False,
                "notes": (
                    f"Archive bytes={archive_size}; extracted tree SHA-256={tree_hash}; "
                    f"COCO8 is a {len(sequence_ids)}-image development subset. Review "
                    "COCO image/annotation terms and the Ultralytics package/model AGPL-3.0 "
                    "license before use; this is not official COCO evaluation."
                ),
            }
        )
        dataset_manifest_path = output_dir / "dataset-manifest.json"
        write_json_atomic(dataset_manifest_path, dataset_manifest.model_dump(mode="json"))
        truth_path = output_dir / "ground-truth.json"
        prediction_path = output_dir / "predictions.json"
        write_json_atomic(
            truth_path,
            {"schema_version": "roadsense.sequence-bundle/v1", "sequences": truth_sequences},
        )
        write_json_atomic(
            prediction_path,
            {"schema_version": "roadsense.sequence-bundle/v1", "sequences": prediction_sequences},
        )
        spec = {
            "schema_version": "roadsense.local-evaluation/v1",
            "dataset_manifest": dataset_manifest_path.name,
            "split": args.split,
            "split_sequences": {args.split: sequence_ids},
            "tasks": ["detection"],
            "ground_truth": truth_path.name,
            "predictions": prediction_path.name,
            "model_artifact": {"manifest": manifest_path.name, "root": artifact_root_ref},
            "protocols": {"detection_iou_threshold": 0.5},
        }
        spec_path = output_dir / "evaluation.json"
        write_json_atomic(spec_path, spec)
        report = evaluate_local(spec_path)
        report_path = output_dir / "report.json"
        write_json_atomic(report_path, report)
        run_material: dict[str, object] = {
            "schema_version": "roadsense.real-evaluation-run/v1",
            "run_mode": "local_experiment",
            "dataset": {
                "source_url": DATASET_SOURCE_URL,
                "archive_sha256": archive_hash,
                "tree_sha256": tree_hash,
                "split": args.split,
                "sequence_ids": sequence_ids,
                "image_count": len(images),
            },
            "model": {
                "manifest_sha256": receipt.manifest_sha256,
                "artifact_sha256": receipt.artifact_sha256,
                "artifact_size_bytes": receipt.artifact_size_bytes,
                "dependency_lock_sha256": receipt.dependency_lock_sha256,
                "artifact_id": receipt.artifact_id,
                "source_url": MODEL_SOURCE_URL,
            },
            "runtime": {
                "python": platform.python_version(),
                "onnxruntime": importlib.metadata.version("onnxruntime"),
                "numpy": importlib.metadata.version("numpy"),
                "providers": session.get_providers(),
                "input_name": input_name,
                "input_shape": list(input_meta.shape),
                "output_name": output_name,
                "output_shape": list(output_meta.shape),
                "preprocessing": {
                    "resize": "letterbox",
                    "size": [INPUT_SIZE, INPUT_SIZE],
                    "interpolation": "bilinear",
                    "padding_anchor": "top_left_integer_floor",
                    "scale": 1.0 / 255.0,
                    "pad_value": 114,
                },
            },
            "inference": {
                "score_threshold": args.score_threshold,
                "nms_iou": args.nms_iou,
                "max_detections": args.max_detections,
                "total_detections": total_detections,
                "per_image_wall_time_ms": inference_ms,
                "total_wall_time_ms": sum(inference_ms),
                "mean_wall_time_ms": float(np.mean(inference_ms)),
            },
            "report_id": report["report_id"],
            "claim_boundary": (
                f"Development evidence on the {len(images)}-image COCO8 {args.split} subset "
                "using an operator-provided local AGPL model. Not official COCO mAP, not a "
                "production FPS claim, and not generalized model quality."
            ),
        }
        run_material["run_id"] = canonical_sha256(run_material)[:16]
        write_json_atomic(output_dir / "run-receipt.json", run_material)
        print(
            json.dumps(
                {
                    "status": "ok",
                    "report": str(report_path),
                    "run_receipt": str(output_dir / "run-receipt.json"),
                    "report_id": report["report_id"],
                    "metrics": report["metrics"],
                    "image_count": len(images),
                    "model_artifact_sha256": receipt.artifact_sha256,
                    "dependency_lock_sha256": receipt.dependency_lock_sha256,
                    "claim_boundary": run_material["claim_boundary"],
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    except (ImportError, OSError, RuntimeError, TypeError, ValueError, MemoryError) as exc:
        print(f"real COCO8 evaluation failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
