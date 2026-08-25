"""Offline contract tests for the BDD100K Detection benchmark runner.

The tests intentionally use synthetic image bytes, a fake ONNX session, and a
temporary evaluator module.  They exercise the protocol boundaries without
downloading BDD100K, loading its labels, or requiring model weights.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

Image = pytest.importorskip("PIL.Image")

from roadsense.contracts import BoxXYXY, Detection
from roadsense.json_io import canonical_sha256, load_strict_json
from scripts.run_bdd100k_detection_benchmark import (
    BDD100K_DETECTION_CATEGORIES,
    COCO_TO_BDD_CATEGORY,
    DEVKIT_MODULE,
    FrozenManifestError,
    _public_evaluator_command,
    _validate_prediction_document,
    build_prediction_document,
    detections_to_bdd_labels,
    freeze_image_manifest,
    load_frozen_image_manifest,
    map_coco_category,
    run_evaluation,
    run_inference,
    validate_frozen_image_manifest,
)


def _manifest(
    tmp_path: Path, names: tuple[str, ...] = ("a.jpg", "b.jpg")
) -> tuple[Path, Path, dict[str, object]]:
    image_root = tmp_path / "images"
    image_root.mkdir()
    records: list[dict[str, object]] = []
    for index, name in enumerate(names):
        image = Image.new("RGB", (32 + index, 24), (index * 40, 20, 10))
        path = image_root / name
        image.save(path, format="JPEG")
        data = path.read_bytes()
        import hashlib

        records.append(
            {"name": name, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
        )
    records.sort(key=lambda item: str(item["name"]))
    raw: dict[str, object] = {
        "schema_version": "roadsense.bdd100k-detection-images/v1",
        "dataset_id": "BDD100K",
        "task": "detection",
        "split": "val",
        "image_count": len(records),
        "images_tree_sha256": canonical_sha256(records),
        "images": records,
    }
    source = tmp_path / "image-manifest.json"
    source.write_text(json.dumps(raw), encoding="utf-8")
    frozen_path = tmp_path / "frozen-image-manifest.json"
    frozen = freeze_image_manifest(source, frozen_path)
    return image_root, frozen_path, frozen


def test_freeze_manifest_binds_images_and_rejects_mutation(tmp_path: Path) -> None:
    image_root, frozen_path, frozen = _manifest(tmp_path)
    assert image_root.is_dir()
    assert frozen["frozen"] is True
    assert load_frozen_image_manifest(frozen_path)["manifest_sha256"] == frozen["manifest_sha256"]

    tampered = json.loads(frozen_path.read_text(encoding="utf-8"))
    tampered["images"][0]["bytes"] += 1
    with pytest.raises(FrozenManifestError, match="manifest_sha256|tree"):
        validate_frozen_image_manifest(tampered)


def test_formal_evaluation_requires_an_explicit_independent_role(tmp_path: Path) -> None:
    records = [
        {"name": f"frame-{index:05d}.jpg", "sha256": "1" * 64, "bytes": 1}
        for index in range(10_000)
    ]
    manifest = freeze_image_manifest(
        {
            "schema_version": "roadsense.bdd100k-detection-images/v1",
            "dataset_id": "BDD100K",
            "task": "detection",
            "split": "val",
            "image_count": len(records),
            "images_tree_sha256": canonical_sha256(records),
            "images": records,
        }
    )
    gt = tmp_path / "gt.json"
    pred = tmp_path / "pred.json"
    gt.write_text("[]", encoding="utf-8")
    pred.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="requires --role independent_a or independent_b"):
        run_evaluation(
            ground_truth=gt,
            predictions=pred,
            output_dir=tmp_path / "evaluation",
            image_manifest=manifest,
        )


def test_mapping_is_explicit_and_never_fabricates_rider_or_sign() -> None:
    assert len(COCO_TO_BDD_CATEGORY) == 80
    assert map_coco_category(0) == "pedestrian"
    assert map_coco_category(11) is None  # stop sign is not traffic sign
    assert map_coco_category(12) is None  # parking meter
    assert "rider" in BDD100K_DETECTION_CATEGORIES
    assert "traffic sign" in BDD100K_DETECTION_CATEGORIES

    detections = [
        Detection(category_id=0, score=0.9, bbox=BoxXYXY(x_min=1, y_min=2, x_max=10, y_max=12)),
        Detection(category_id=11, score=0.8, bbox=BoxXYXY(x_min=2, y_min=3, x_max=11, y_max=13)),
    ]
    labels, dropped = detections_to_bdd_labels(detections)
    assert [label["category"] for label in labels] == ["pedestrian"]
    assert dropped == {"stop sign": 1}


def test_prediction_document_has_empty_frame_for_every_manifest_image() -> None:
    detections = {
        "a.jpg": [
            Detection(category_id=2, score=0.7, bbox=BoxXYXY(x_min=1, y_min=2, x_max=9, y_max=12))
        ]
    }
    frames, metadata = build_prediction_document(["a.jpg", "b.jpg"], detections)
    assert [frame["name"] for frame in frames] == ["a.jpg", "b.jpg"]
    assert frames[1]["labels"] == []
    assert metadata["frame_count"] == 2
    assert metadata["empty_frame_count"] == 1

    with pytest.raises(ValueError, match="exactly and canonically"):
        _validate_prediction_document(frames, expected_names=["b.jpg", "a.jpg"])


def test_public_evaluator_command_redacts_both_path_styles() -> None:
    command = [
        r"C:\\private\\venv\\python.exe",
        "-m",
        DEVKIT_MODULE,
        "-t",
        "det",
        "-g",
        r"C:\\private\\gt.json",
        "-r",
        "/private/pred.json",
        "--out-file",
        r"C:\\private\\result.json",
    ]
    public = _public_evaluator_command(command)
    assert public == [
        "python.exe",
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


class _FakeSession:
    def __init__(self, output: np.ndarray) -> None:
        self.output = output

    def get_inputs(self) -> list[object]:
        return [SimpleNamespace(name="images", type="tensor(float)", shape=["batch", 3, 640, 640])]

    def get_outputs(self) -> list[object]:
        return [
            SimpleNamespace(name="output0", type="tensor(float)", shape=["batch", 84, "anchors"])
        ]

    def get_providers(self) -> list[str]:
        return ["CPUExecutionProvider"]

    def run(self, names: list[str], feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
        assert names == ["output0"]
        assert feeds["images"].shape == (1, 3, 640, 640)
        return [self.output]


def test_inference_never_reads_labels_and_writes_complete_scalabel_json(tmp_path: Path) -> None:
    image_root, manifest_path, frozen = _manifest(tmp_path)
    # A COCO class 2 (car) candidate and class 11 (stop sign) candidate.  The
    # latter must be counted as dropped instead of becoming traffic sign.
    output = np.zeros((1, 84, 2), dtype=np.float32)
    output[0, 0:4, 0] = (16, 92, 10, 8)
    output[0, 4 + 2, 0] = 0.9
    output[0, 0:4, 1] = (20, 95, 8, 6)
    output[0, 4 + 11, 1] = 0.8
    model = tmp_path / "model.onnx"
    model.write_bytes(b"synthetic-model")
    output_dir = tmp_path / "run"

    receipt = run_inference(
        model=model,
        image_manifest=manifest_path,
        images_root=image_root,
        output_dir=output_dir,
        session=_FakeSession(output),
    )
    prediction = load_strict_json(output_dir / "predictions.json")
    assert len(prediction) == 2
    assert [frame["name"] for frame in prediction] == ["a.jpg", "b.jpg"]
    assert all("labels" in frame for frame in prediction)
    assert receipt["dataset"]["image_manifest_sha256"] == frozen["manifest_sha256"]
    assert receipt["ontology"]["dropped_coco_detections"] == {"stop sign": 2}
    assert (output_dir / "runtime-lock.txt").is_file()


def test_inference_requires_frozen_manifest(tmp_path: Path) -> None:
    image_root, _frozen_path, frozen = _manifest(tmp_path)
    mutable = dict(frozen)
    mutable["frozen"] = False
    model = tmp_path / "model.onnx"
    model.write_bytes(b"model")
    with pytest.raises(FrozenManifestError, match="frozen image manifest"):
        run_inference(
            model=model,
            image_manifest=mutable,
            images_root=image_root,
            output_dir=tmp_path / "run",
            session=_FakeSession(np.zeros((1, 84, 1), dtype=np.float32)),
        )


def test_evaluate_invokes_isolated_devkit_and_records_stream_hashes(tmp_path: Path) -> None:
    _image_root, manifest_path, _frozen = _manifest(tmp_path, names=("a.jpg",))
    gt = tmp_path / "gt.json"
    gt.write_text("[]", encoding="utf-8")
    prediction = tmp_path / "pred.json"
    prediction.write_text(
        json.dumps([{"name": "a.jpg", "labels": [], "attributes": {}}]), encoding="utf-8"
    )

    # Build a tiny isolated bdd100k.eval.run package.  The runner still uses
    # the exact official command shape and captures stdout/stderr externally.
    package = tmp_path / "bdd100k" / "eval"
    package.mkdir(parents=True)
    (tmp_path / "bdd100k" / "__init__.py").write_text("", encoding="utf-8")
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "run.py").write_text(
        "import json\nprint(json.dumps({'mAP': 0.0, 'protocol': 'det'}))\n", encoding="utf-8"
    )
    receipt = run_evaluation(
        ground_truth=gt,
        predictions=prediction,
        output_dir=tmp_path / "eval",
        evaluator_python=sys.executable,
        evaluator_cwd=tmp_path,
        image_manifest=manifest_path,
    )
    assert receipt["status"] == "ok"
    evaluator = receipt["evaluator"]
    assert evaluator["command"][1:4] == ["-m", DEVKIT_MODULE, "-t"]
    assert evaluator["command"][-5:] == ["<GT>", "-r", "<PRED>", "--out-file", "<RESULT>"]
    # The public receipt must not disclose the temporary workspace used by
    # this test (or by a real local benchmark run).
    receipt_text = json.dumps(receipt, ensure_ascii=False)
    assert str(tmp_path) not in receipt_text
    assert evaluator["stdout_sha256"]
    assert evaluator["stderr_sha256"]
    assert evaluator["result_sha256"]
    assert evaluator["result_source"] == "stdout_fallback"
    assert evaluator["result"]["path"] == "evaluator-result.stdout-fallback.json"
    assert (tmp_path / "eval" / evaluator["result"]["path"]).is_file()
    assert (tmp_path / "eval" / "evaluation-receipt.json").is_file()


def test_evaluate_rejects_successful_wrapper_without_result_evidence(tmp_path: Path) -> None:
    """A zero exit code without JSON output is not a benchmark result."""

    _image_root, manifest_path, _frozen = _manifest(tmp_path, names=("a.jpg",))
    gt = tmp_path / "gt.json"
    gt.write_text("[]", encoding="utf-8")
    prediction = tmp_path / "pred.json"
    prediction.write_text(
        json.dumps([{"name": "a.jpg", "labels": [], "attributes": {}}]), encoding="utf-8"
    )
    package = tmp_path / "bdd100k" / "eval"
    package.mkdir(parents=True)
    (tmp_path / "bdd100k" / "__init__.py").write_text("", encoding="utf-8")
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "run.py").write_text("# intentionally emits no result\n", encoding="utf-8")
    (tmp_path / "eval").mkdir()
    # A stale output from a previous invocation must not be accepted as this
    # run's evaluator result.
    (tmp_path / "eval" / "evaluator-result.json").write_text(
        json.dumps({"mAP": 1.0}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="devkit failed"):
        run_evaluation(
            ground_truth=gt,
            predictions=prediction,
            output_dir=tmp_path / "eval",
            evaluator_python=sys.executable,
            evaluator_cwd=tmp_path,
            image_manifest=manifest_path,
        )
    receipt = load_strict_json(tmp_path / "eval" / "evaluation-receipt.json")
    evaluator = receipt["evaluator"]
    assert receipt["status"] == "failed"
    assert evaluator["result_source"] == "stdout_fallback"
    result_path = tmp_path / "eval" / evaluator["result"]["path"]
    assert result_path.is_file()
    assert evaluator["result"]["bytes"] == result_path.stat().st_size


def test_evaluate_rejects_output_collision_with_inputs(tmp_path: Path) -> None:
    _image_root, manifest_path, _frozen = _manifest(tmp_path, names=("a.jpg",))
    output_dir = tmp_path / "eval"
    output_dir.mkdir()
    gt = output_dir / "evaluator-result.json"
    gt.write_text("[]", encoding="utf-8")
    prediction = tmp_path / "pred.json"
    prediction.write_text(
        json.dumps([{"name": "a.jpg", "labels": [], "attributes": {}}]), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="must not overwrite"):
        run_evaluation(
            ground_truth=gt,
            predictions=prediction,
            output_dir=output_dir,
            evaluator_python=sys.executable,
            image_manifest=manifest_path,
        )


def test_evaluate_accepts_pretty_json_stdout_fallback(tmp_path: Path) -> None:
    _image_root, manifest_path, _frozen = _manifest(tmp_path, names=("a.jpg",))
    gt = tmp_path / "gt.json"
    gt.write_text("[]", encoding="utf-8")
    prediction = tmp_path / "pred.json"
    prediction.write_text(
        json.dumps([{"name": "a.jpg", "labels": [], "attributes": {}}]), encoding="utf-8"
    )
    package = tmp_path / "bdd100k" / "eval"
    package.mkdir(parents=True)
    (tmp_path / "bdd100k" / "__init__.py").write_text("", encoding="utf-8")
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "run.py").write_text(
        "import json; print(json.dumps({'mAP': 0.25}, indent=2))\n", encoding="utf-8"
    )
    receipt = run_evaluation(
        ground_truth=gt,
        predictions=prediction,
        output_dir=tmp_path / "eval",
        evaluator_python=sys.executable,
        evaluator_cwd=tmp_path,
        image_manifest=manifest_path,
    )
    assert receipt["status"] == "ok"
    assert receipt["evaluator"]["metrics"] == {"mAP": 0.25}
    artifact = tmp_path / "eval" / receipt["evaluator"]["result"]["path"]
    assert artifact.read_text(encoding="utf-8").lstrip().startswith("{")


def test_evaluate_does_not_follow_preexisting_stream_symlink(tmp_path: Path) -> None:
    _image_root, manifest_path, _frozen = _manifest(tmp_path, names=("a.jpg",))
    gt = tmp_path / "gt.json"
    gt.write_text("[]", encoding="utf-8")
    prediction = tmp_path / "pred.json"
    prediction.write_text(
        json.dumps([{"name": "a.jpg", "labels": [], "attributes": {}}]), encoding="utf-8"
    )
    package = tmp_path / "bdd100k" / "eval"
    package.mkdir(parents=True)
    (tmp_path / "bdd100k" / "__init__.py").write_text("", encoding="utf-8")
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "run.py").write_text(
        "import json; print(json.dumps({'mAP': 0.0}))\n", encoding="utf-8"
    )
    output_dir = tmp_path / "eval"
    output_dir.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("must remain unchanged", encoding="utf-8")
    stream = output_dir / "evaluator.stdout.txt"
    try:
        stream.symlink_to(outside)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")
    with pytest.raises(ValueError, match="must not be a symlink"):
        run_evaluation(
            ground_truth=gt,
            predictions=prediction,
            output_dir=output_dir,
            evaluator_python=sys.executable,
            evaluator_cwd=tmp_path,
            image_manifest=manifest_path,
        )
    assert outside.read_text(encoding="utf-8") == "must remain unchanged"
