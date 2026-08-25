"""Offline tests for the sanitized BDD100K benchmark finalizer."""

from __future__ import annotations

from copy import deepcopy

import pytest

from roadsense.json_io import canonical_sha256
from scripts.finalize_bdd100k_detection_benchmark import (
    BDD100KFinalizeError,
    compute_archive_sha256,
    finalize_bdd100k_detection_benchmark,
)
from scripts.run_bdd100k_detection_benchmark import freeze_image_manifest


def _sha(char: str) -> str:
    return char * 64


def _evidence() -> dict[str, object]:
    archives = [
        {"role": "images_val_zip", "format": "zip", "sha256": _sha("1"), "bytes": 11},
        {"role": "det_20_labels", "format": "zip", "sha256": _sha("2"), "bytes": 22},
    ]
    image_records = [{"name": "a.jpg", "sha256": _sha("3"), "bytes": 1}]
    frozen_image = freeze_image_manifest(
        {
            "schema_version": "roadsense.bdd100k-detection-images/v1",
            "dataset_id": "BDD100K",
            "task": "detection",
            "split": "val",
            "image_count": 1,
            "images_tree_sha256": canonical_sha256(image_records),
            "images": image_records,
            "source_archives": archives,
        }
    )
    labels_sha = _sha("4")
    content_sha = canonical_sha256(
        {
            "dataset_id": "BDD100K",
            "task": "detection",
            "split": "val",
            "images_tree_sha256": frozen_image["images_tree_sha256"],
            "labels_sha256": labels_sha,
            "image_count": 1,
        }
    )
    source = {
        "schema_version": "roadsense.bdd100k-detection-source-receipt/v1",
        "dataset_id": "BDD100K",
        "task": "detection",
        "split": "val",
        "local_only": True,
        "license_acceptance": {"accepted": True},
        "source_archives": archives,
        "images_tree_sha256": frozen_image["images_tree_sha256"],
        "labels_sha256": labels_sha,
        "labels_bytes": 1,
        "content_sha256": content_sha,
    }
    dataset = {
        "schema_version": "roadsense.dataset-manifest/v1",
        "dataset_name": "BDD100K Detection 2020 validation",
        "source_url": "https://bdd-data.berkeley.edu/",
        "license_id": "BDD100K-research-license-accepted-locally",
        "tasks": ["detection"],
        "splits": {"val": "1 image"},
        "content_sha256": content_sha,
        "evaluation_authorized": True,
        "frozen": False,
        "notes": "synthetic",
    }
    split = {
        "schema_version": "roadsense.bdd100k-detection-split-inventory/v1",
        "dataset_id": "BDD100K",
        "task": "detection",
        "split": "val",
        "image_count": 1,
        "images_tree_sha256": frozen_image["images_tree_sha256"],
        "labels": {"sha256": labels_sha, "bytes": 1, "frame_count": 1},
    }
    mapping = {"person": "pedestrian", "stop sign": None}
    model = {
        "schema_version": "roadsense.bdd100k-detection-model/v1",
        "model_id": "ultralytics-yolo11n-coco-onnx",
        "artifact_sha256": _sha("5"),
        "artifact_bytes": 5,
        "artifact_format": "onnx",
        "framework": "ultralytics",
        "framework_version": "8.3.237",
        "runtime": "onnxruntime",
        "input": {"size": [640, 640], "layout": "NCHW", "color_order": "RGB"},
        "tasks": ["detection"],
        "source": "https://example.invalid/model",
        "license_id": "AGPL-3.0",
        "ontology": {"source": "COCO-80", "target": ["pedestrian"], "mapping": mapping},
        "claim_boundary": "synthetic",
    }
    model_manifest_sha = canonical_sha256(model)
    prediction_sha = _sha("6")
    config_sha = _sha("7")
    inference = {
        "schema_version": "roadsense.bdd100k-detection-inference/v1",
        "stage": "infer",
        "dataset": {
            "dataset_id": "BDD100K",
            "task": "detection",
            "release": "2020",
            "split": "val",
            "image_manifest_sha256": frozen_image["manifest_sha256"],
            "images_tree_sha256": frozen_image["images_tree_sha256"],
            "image_count": 1,
        },
        "model": {
            "path_name": "model.onnx",
            "sha256": model["artifact_sha256"],
            "bytes": model["artifact_bytes"],
            "input_size": [640, 640],
            "manifest_sha256": model_manifest_sha,
        },
        "runtime": {"runtime_lock_sha256": _sha("8")},
        "inference": {"config_sha256": config_sha},
        "ontology": {"mapping_sha256": canonical_sha256(mapping)},
        "prediction": {
            "path": "predictions.json",
            "sha256": prediction_sha,
            "bytes": 2,
            "canonical_sha256": _sha("9"),
            "schema_version": "scalabel.bdd100k-detection/v1",
        },
    }

    def evaluation(run_id: str) -> dict[str, object]:
        result_sha = _sha("e") if run_id == "run-a" else _sha("f")
        return {
            "schema_version": "roadsense.bdd100k-detection-evaluation/v1",
            "stage": "evaluate",
            "status": "ok",
            "dataset": {
                "ground_truth_sha256": labels_sha,
                "split": "val",
                "image_manifest_sha256": frozen_image["manifest_sha256"],
            },
            "prediction": {"sha256": prediction_sha, "bytes": 2, "frame_count": 1},
            "evaluator": {
                "id": "bdd100k-devkit",
                "module": "bdd100k.eval.run",
                "commit": "9ac17c6c7c51d2fc83065fccd707cd5b1882a293",
                "command": ["python", "-m", "bdd100k.eval.run", "-t", "det"],
                "returncode": 0,
                "timed_out": False,
                "runtime_lock_sha256": _sha("a"),
                "evaluator_config_sha256": _sha("b"),
                "packages": {"bdd100k": "0.1.0"},
                "stdout_sha256": _sha("c"),
                "stderr_sha256": _sha("d"),
                "result_sha256": result_sha,
                "metrics": {"AP": 0.321, "AP50": 0.5},
                "result": {"path": "evaluator-result.json", "sha256": result_sha},
            },
            "run_id": run_id,
        }

    return {
        "source_receipt": source,
        "dataset_manifest": dataset,
        "split_inventory": split,
        "image_manifest": frozen_image,
        "model_manifest": model,
        "inference_receipt": inference,
        "evaluation_a": evaluation("run-a"),
        "evaluation_b": evaluation("run-b"),
    }


def test_archive_hash_is_stable_for_reduced_contract_evidence() -> None:
    evidence = _evidence()
    archive_hash = compute_archive_sha256(evidence["source_receipt"])  # type: ignore[arg-type]
    assert len(archive_hash) == 64
    assert archive_hash == compute_archive_sha256(evidence["source_receipt"])  # type: ignore[arg-type]


def test_finalize_rejects_metric_or_prediction_disagreement() -> None:
    evidence = _evidence()
    changed = deepcopy(evidence["evaluation_b"])
    assert isinstance(changed, dict)
    evaluator = changed["evaluator"]
    assert isinstance(evaluator, dict)
    evaluator["metrics"] = {"AP": 0.322, "AP50": 0.5}
    evidence["evaluation_b"] = changed
    with pytest.raises(BDD100KFinalizeError, match="identical metrics"):
        finalize_bdd100k_detection_benchmark(**evidence)  # type: ignore[arg-type]

    evidence = _evidence()
    changed = deepcopy(evidence["evaluation_b"])
    assert isinstance(changed, dict)
    changed["prediction"] = {"sha256": _sha("9"), "bytes": 2, "frame_count": 1}
    evidence["evaluation_b"] = changed
    with pytest.raises(BDD100KFinalizeError, match="prediction hash"):
        finalize_bdd100k_detection_benchmark(**evidence)  # type: ignore[arg-type]


def test_finalize_rejects_unpinned_devkit_and_nonfinite_metrics() -> None:
    evidence = _evidence()
    changed = deepcopy(evidence["evaluation_a"])
    assert isinstance(changed, dict)
    evaluator = changed["evaluator"]
    assert isinstance(evaluator, dict)
    evaluator["commit"] = "0" * 40
    evidence["evaluation_a"] = changed
    with pytest.raises(BDD100KFinalizeError, match="unpinned"):
        finalize_bdd100k_detection_benchmark(**evidence)  # type: ignore[arg-type]

    evidence = _evidence()
    changed = deepcopy(evidence["evaluation_b"])
    assert isinstance(changed, dict)
    evaluator = changed["evaluator"]
    assert isinstance(evaluator, dict)
    evaluator["metrics"] = {"AP": float("nan")}
    evidence["evaluation_b"] = changed
    with pytest.raises(BDD100KFinalizeError, match="finite"):
        finalize_bdd100k_detection_benchmark(**evidence)  # type: ignore[arg-type]
