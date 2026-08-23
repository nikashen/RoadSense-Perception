from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from roadsense.adapters import MODEL_ARTIFACT_SCHEMA
from roadsense.cli import main
from roadsense.evidence import compute_report_id
from roadsense.local_eval import LocalEvaluationError, evaluate_local, load_local_spec


def _frame(*, index: int, track_id: int = 1, category_id: int = 1) -> dict[str, object]:
    return {
        "frame_index": index,
        "timestamp_ms": index * 100,
        "image_size": {"width": 20, "height": 20},
        "detections": [
            {
                "category_id": category_id,
                "score": 0.9,
                "track_id": track_id,
                "bbox": {"x_min": 2, "y_min": 2, "x_max": 10, "y_max": 10},
            }
        ],
    }


def _write_json(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_inputs(tmp_path: Path, *, tasks: list[str] | None = None) -> Path:
    selected_tasks = tasks or ["detection", "tracking"]
    manifest = _write_json(
        tmp_path / "dataset-manifest.json",
        {
            "schema_version": "roadsense.dataset-manifest/v1",
            "dataset_name": "operator-local-test",
            "source_url": "operator://local-test",
            "license_id": "local-test-license",
            "tasks": selected_tasks,
            "splits": {"development": "seq-a", "final": "seq-b"},
            "content_sha256": "1" * 64,
            "evaluation_authorized": False,
            "frozen": False,
        },
    )
    sequence_bundle = {
        "schema_version": "roadsense.sequence-bundle/v1",
        "sequences": [{"sequence_id": "seq-a", "frames": [_frame(index=0), _frame(index=1)]}],
    }
    truth = _write_json(tmp_path / "truth.json", sequence_bundle)
    prediction = _write_json(tmp_path / "prediction.json", sequence_bundle)
    spec_payload: dict[str, object] = {
        "schema_version": "roadsense.local-evaluation/v1",
        "dataset_manifest": manifest.name,
        "split": "development",
        "split_sequences": {"development": ["seq-a"], "final": ["seq-b"]},
        "tasks": selected_tasks,
        "ground_truth": truth.name,
        "predictions": prediction.name,
    }
    if "segmentation" in selected_tasks:
        truth_mask = tmp_path / "truth-seq-a.npy"
        prediction_mask = tmp_path / "prediction-seq-a.npy"
        np.save(truth_mask, np.zeros((2, 4, 4), dtype=np.int64))
        np.save(prediction_mask, np.zeros((2, 4, 4), dtype=np.int64))
        spec_payload["segmentation"] = {
            "ground_truth": {"seq-a": truth_mask.name},
            "predictions": {"seq-a": prediction_mask.name},
            "num_classes": 2,
            "ignore_index": 255,
        }
    return _write_json(tmp_path / "evaluation.json", spec_payload)


def test_fixture_dry_run_is_explicitly_non_benchmark(capsys) -> None:
    assert main(["evaluate-local", "--fixture", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["evidence_level"] == "fixture"
    assert payload["evaluation_authorized"] is False
    assert payload["frozen"] is False
    assert payload["benchmark_claim_available"] is False


def test_local_detection_and_tracking_report_is_development_evidence(tmp_path: Path) -> None:
    spec = _write_inputs(tmp_path)
    report = evaluate_local(spec)
    assert report["evidence_level"] == "development"
    assert report["evaluation_authorized"] is False
    assert report["frozen"] is False
    assert report["metrics"] == {
        "detection_ap": 1.0,
        "detection_precision": 1.0,
        "detection_recall": 1.0,
        "tracking_identity_f1": 1.0,
        "tracking_mota": 1.0,
    }
    assert report["report_id"]
    assert report["report_id"] == compute_report_id(report)
    assert report["details"]["sequence_ids"] == ["seq-a"]  # type: ignore[index]
    assert report["details"]["benchmark_claim_available"] is False  # type: ignore[index]


def test_local_segmentation_report_uses_npy_masks(tmp_path: Path) -> None:
    spec = _write_inputs(tmp_path, tasks=["segmentation"])
    report = evaluate_local(spec)
    assert report["metrics"] == {
        "segmentation_mean_iou": 1.0,
        "segmentation_pixel_accuracy": 1.0,
    }


def test_local_spec_rejects_network_paths_without_downloading(tmp_path: Path) -> None:
    spec = _write_inputs(tmp_path)
    payload = json.loads(spec.read_text(encoding="utf-8"))
    payload["ground_truth"] = "https://example.invalid/truth.json"
    spec.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(LocalEvaluationError, match="never downloaded"):
        load_local_spec(spec)


def test_local_spec_rejects_overlapping_sequence_splits(tmp_path: Path) -> None:
    spec = _write_inputs(tmp_path)
    payload = json.loads(spec.read_text(encoding="utf-8"))
    payload["split_sequences"]["final"] = ["seq-a"]
    spec.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(LocalEvaluationError, match="disjoint"):
        load_local_spec(spec)


def test_local_evaluation_fails_closed_when_bundle_has_extra_sequence(tmp_path: Path) -> None:
    spec = _write_inputs(tmp_path)
    prediction_path = tmp_path / "prediction.json"
    payload = json.loads(prediction_path.read_text(encoding="utf-8"))
    payload["sequences"].append({"sequence_id": "seq-extra", "frames": [_frame(index=0)]})
    prediction_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(LocalEvaluationError, match="exactly the selected split sequences"):
        evaluate_local(spec)


def test_local_detection_rejects_empty_ground_truth(tmp_path: Path) -> None:
    spec = _write_inputs(tmp_path, tasks=["detection"])
    truth_path = tmp_path / "truth.json"
    payload = json.loads(truth_path.read_text(encoding="utf-8"))
    payload["sequences"][0]["frames"][0]["detections"] = []
    payload["sequences"][0]["frames"][1]["detections"] = []
    truth_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(LocalEvaluationError, match="undefined per-sequence AP"):
        evaluate_local(spec)


def test_local_report_binds_verified_model_artifact(tmp_path: Path) -> None:
    spec = _write_inputs(tmp_path)
    artifact_root = tmp_path / "artifacts"
    artifact_path = artifact_root / "weights" / "model.onnx"
    artifact_path.parent.mkdir(parents=True)
    artifact_bytes = b"local-eval-model-artifact"
    artifact_path.write_bytes(artifact_bytes)
    manifest_payload = {
        "schema_version": MODEL_ARTIFACT_SCHEMA,
        "artifact_id": "local-detector-v1",
        "artifact_path": "weights/model.onnx",
        "artifact_sha256": hashlib.sha256(artifact_bytes).hexdigest(),
        "artifact_size_bytes": len(artifact_bytes),
        "artifact_format": "onnx",
        "framework": "test",
        "framework_version": "1.0",
        "backend": "test-runtime",
        "backend_version": "1.0",
        "tasks": ["detection", "tracking"],
        "ontology": [{"category_id": 1, "label": "car"}],
        "input": {
            "width": 20,
            "height": 20,
            "channels": 3,
            "dtype": "float32",
            "layout": "NCHW",
            "color_order": "RGB",
            "coordinate_space": "original_xyxy",
        },
        "output": {
            "output_schema": "roadsense.detections/v1",
            "coordinate_space": "original_xyxy",
            "score_semantics": "class_probability",
        },
        "preprocessing": {"resize_mode": "none", "scale": 1.0},
        "runtime": {
            "device": "cpu",
            "precision": "fp32",
            "runtime_name": "test",
            "runtime_version": "1.0",
            "deterministic": True,
        },
        "license_id": "local-test",
        "source": "operator://local-artifact",
    }
    manifest_path = artifact_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest_payload), encoding="utf-8")
    payload = json.loads(spec.read_text(encoding="utf-8"))
    payload["model_artifact"] = {
        "manifest": "artifacts/manifest.json",
        "root": "artifacts",
    }
    spec.write_text(json.dumps(payload), encoding="utf-8")

    report = evaluate_local(spec)
    artifact_details = report["details"]["model_artifact"]  # type: ignore[index]
    assert artifact_details["bound"] is True  # type: ignore[index]
    assert artifact_details["artifact_id"] == "local-detector-v1"  # type: ignore[index]
    assert artifact_details["artifact_sha256"] == hashlib.sha256(artifact_bytes).hexdigest()  # type: ignore[index]
    assert "model_artifact_manifest" in report["details"]["input_hashes"]  # type: ignore[index]
    assert report["report_id"] == compute_report_id(report)


def test_authorized_local_evaluation_requires_model_artifact(tmp_path: Path) -> None:
    spec = _write_inputs(tmp_path)
    manifest_path = tmp_path / "dataset-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["evaluation_authorized"] = True
    manifest["frozen"] = True
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(LocalEvaluationError, match="requires a verified model_artifact"):
        evaluate_local(spec)


def test_cli_reports_missing_local_spec_explainably(capsys) -> None:
    assert main(["evaluate-local"]) == 2
    assert "local evaluation spec is required" in capsys.readouterr().err
