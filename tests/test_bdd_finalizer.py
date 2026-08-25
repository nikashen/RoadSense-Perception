"""Path-backed contract tests for the BDD100K benchmark finalizer.

These tests intentionally do not contain BDD media, labels, weights, or an
evaluator checkout.  They synthesize the receipts emitted by the preparation,
freeze, inference, and evaluator stages, then materialize only the small files
whose hashes the finalizer is expected to verify.  This exercises the release
boundary without making a benchmark claim from synthetic data.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

import scripts.finalize_bdd100k_detection_benchmark as finalizer_module
from roadsense.json_io import canonical_sha256, load_strict_json
from scripts.finalize_bdd100k_detection_benchmark import (
    BDD100KFinalizeError,
    _validate_evaluator_receipt,
    _validate_official_source_attestation,
    finalize_bdd100k_detection_benchmark,
)
from scripts.run_bdd100k_detection_benchmark import freeze_image_manifest
from scripts.verify_bdd100k_detection_benchmark import verify_bdd100k_detection_benchmark

DEVKIT_COMMIT = "9ac17c6c7c51d2fc83065fccd707cd5b1882a293"


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_tag(tag: str) -> str:
    """Return a valid deterministic digest for a synthetic receipt field."""

    return _sha_bytes(tag.encode("ascii"))


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def _synthetic_evidence(tmp_path: Path, *, formal: bool = False) -> dict[str, Path]:
    """Build path-backed evidence with real or reduced protocol cardinality.

    ``formal=True`` still uses synthetic hashes/bytes and never ships BDD
    media. It only exercises the exact 10,000-image publication gate.
    """

    image_count = finalizer_module.BDD100K_OFFICIAL_IMAGE_COUNT if formal else 1

    archives = [
        {
            "role": "images_val_zip",
            "format": "zip",
            "filename": "bdd100k_images_100k.zip",
            "sha256": _sha_tag("images archive"),
            "bytes": 101,
            "official_source_url": "https://bdd-data.berkeley.edu/images",
            **(
                {"official_package_md5": finalizer_module.BDD100K_OFFICIAL_IMAGES_MD5}
                if formal
                else {}
            ),
        },
        {
            "role": "det_20_labels",
            "format": "zip",
            "filename": "bdd100k_det_20_labels.zip",
            "sha256": _sha_tag("labels archive"),
            "bytes": 202,
            "official_source_url": "https://bdd-data.berkeley.edu/labels",
            **(
                {"official_package_md5": finalizer_module.BDD100K_OFFICIAL_LABELS_MD5}
                if formal
                else {}
            ),
        },
    ]
    image_records = [
        {
            "name": f"frame-{index:05d}.jpg",
            "sha256": _sha_tag(f"frame bytes {index}"),
            "bytes": 10,
        }
        for index in range(image_count)
    ]
    image_manifest = freeze_image_manifest(
        {
            "schema_version": "roadsense.bdd100k-detection-images/v1",
            "dataset_id": "BDD100K",
            "task": "detection",
            "split": "val",
            "image_count": image_count,
            "images_tree_sha256": canonical_sha256(image_records),
            "images": image_records,
            "source_archives": archives,
        }
    )

    labels_sha = _sha_tag("det_val labels")
    content_sha = canonical_sha256(
        {
            "dataset_id": "BDD100K",
            "task": "detection",
            "split": "val",
            "images_tree_sha256": image_manifest["images_tree_sha256"],
            "labels_sha256": labels_sha,
            "image_count": image_count,
        }
    )
    source_receipt: dict[str, Any] = {
        "schema_version": "roadsense.bdd100k-detection-source-receipt/v1",
        "dataset_id": "BDD100K",
        "task": "detection",
        "split": "val",
        "local_only": True,
        "license_acceptance": {
            "accepted": True,
            "flag": "--accept-bdd100k-research-license",
        },
        "source_archives": archives,
        "prepared_layout": {
            "images": "images/val",
            "labels": "labels/det_val.json",
            "image_manifest": "image-manifest.json",
            "split_inventory": "split-inventory.json",
            "dataset_manifest": "dataset-manifest.json",
        },
        "images_tree_sha256": image_manifest["images_tree_sha256"],
        "labels_sha256": labels_sha,
        "labels_bytes": 303,
        "content_sha256": content_sha,
    }
    dataset_manifest: dict[str, Any] = {
        "schema_version": "roadsense.dataset-manifest/v1",
        "dataset_name": "BDD100K Detection 2020 validation",
        "source_url": "https://bdd-data.berkeley.edu/",
        "license_id": "BDD100K-research-license-accepted-locally",
        "tasks": ["detection"],
        "splits": {"val": f"{image_count} images; det_20/det_val.json"},
        "content_sha256": content_sha,
        "evaluation_authorized": True,
        "frozen": False,
        "notes": "Synthetic receipt contract test; no BDD bytes are included.",
    }
    split_inventory: dict[str, Any] = {
        "schema_version": "roadsense.bdd100k-detection-split-inventory/v1",
        "dataset_id": "BDD100K",
        "task": "detection",
        "split": "val",
        "image_directory": "images/val",
        "image_manifest": "image-manifest.json",
        "image_count": image_count,
        "images_tree_sha256": image_manifest["images_tree_sha256"],
        "labels": {
            "path": "labels/det_val.json",
            "sha256": labels_sha,
            "bytes": 303,
            "frame_count": image_count,
            "categories": [
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
            ],
        },
    }

    mapping = {"person": "pedestrian", "stop sign": None}
    model_manifest: dict[str, Any] = {
        "schema_version": "roadsense.bdd100k-detection-model/v1",
        "model_id": "synthetic-model-v1",
        "artifact_sha256": _sha_tag("model artifact"),
        "artifact_bytes": 404,
        "artifact_format": "onnx",
        "framework": "test",
        "framework_version": "1.0",
        "runtime": "onnxruntime",
        "input": {"size": [640, 640], "layout": "NCHW", "color_order": "RGB"},
        "tasks": ["detection"],
        "source": "https://example.invalid/model.onnx",
        "license_id": "MIT-test",
        "ontology": {
            "source": "COCO-80",
            "target": ["pedestrian", "car"],
            "mapping": mapping,
        },
        "claim_boundary": "Synthetic contract test only.",
    }
    model_manifest_sha = canonical_sha256(model_manifest)
    config_sha = _sha_tag("inference config")

    inference_dir = tmp_path / "inference-run"
    inference_dir.mkdir(exist_ok=True)
    prediction_file = inference_dir / "predictions.json"
    prediction_file.write_bytes(b'[{"name":"frame-00000.jpg","labels":[],"attributes":{}}]\n')
    prediction_sha = _sha_bytes(prediction_file.read_bytes())
    inference_receipt: dict[str, Any] = {
        "schema_version": "roadsense.bdd100k-detection-inference/v1",
        "stage": "infer",
        "dataset": {
            "dataset_id": "BDD100K",
            "task": "detection",
            "release": "2020",
            "split": "val",
            "image_manifest_sha256": image_manifest["manifest_sha256"],
            "images_tree_sha256": image_manifest["images_tree_sha256"],
            "image_count": image_count,
        },
        "model": {
            "path_name": "weights/model.onnx",
            "sha256": model_manifest["artifact_sha256"],
            "bytes": model_manifest["artifact_bytes"],
            "input_size": [640, 640],
            "manifest_sha256": model_manifest_sha,
        },
        "runtime": {"runtime_lock_sha256": _sha_tag("inference runtime lock")},
        "inference": {"config_sha256": config_sha},
        "ontology": {"mapping_sha256": canonical_sha256(mapping)},
        "prediction": {
            "path": prediction_file.name,
            "sha256": prediction_sha,
            "canonical_sha256": _sha_tag("prediction canonical"),
            "bytes": prediction_file.stat().st_size,
            "frame_count": image_count,
            "schema_version": "scalabel.bdd100k-detection/v1",
        },
    }

    evaluation_paths: dict[str, Path] = {}
    for role, result_payload in (
        ("evaluation-a", b'{"AP":0.42}\n'),
        ("evaluation-b", b'{ "AP": 0.42 }\n'),
    ):
        evaluation_dir = tmp_path / role
        evaluation_dir.mkdir()
        result_file = evaluation_dir / "evaluator-result.json"
        stdout_file = evaluation_dir / "evaluator.stdout.txt"
        stderr_file = evaluation_dir / "evaluator.stderr.txt"
        result_file.write_bytes(result_payload)
        stdout_file.write_bytes(b"official evaluator stdout\n")
        stderr_file.write_bytes(b"\n")
        result_sha = _sha_bytes(result_file.read_bytes())
        evaluation_receipt: dict[str, Any] = {
            "schema_version": "roadsense.bdd100k-detection-evaluation/v1",
            "stage": "evaluate",
            "status": "ok",
            "role": "independent_a" if role == "evaluation-a" else "independent_b",
            "dataset": {
                "ground_truth_sha256": labels_sha,
                "split": "val",
                "image_manifest_sha256": image_manifest["manifest_sha256"],
            },
            "prediction": {
                "sha256": prediction_sha,
                "bytes": prediction_file.stat().st_size,
                "frame_count": image_count,
            },
            "evaluator": {
                "id": "bdd100k-devkit",
                "module": "bdd100k.eval.run",
                "commit": DEVKIT_COMMIT,
                "runtime_lock_sha256": _sha_tag("evaluator lock"),
                "evaluator_config_sha256": _sha_tag("evaluator config"),
                "packages": {
                    **(
                        finalizer_module.BDD100K_REQUIRED_EVALUATOR_PACKAGES
                        if formal
                        else {"bdd100k": "pinned-test", "scalabel": "pinned-test"}
                    ),
                },
                "returncode": 0,
                "timed_out": False,
                "result_source": "file",
                "stdout_sha256": _sha_bytes(stdout_file.read_bytes()),
                "stderr_sha256": _sha_bytes(stderr_file.read_bytes()),
                "result_sha256": result_sha,
                "result": {
                    "path": result_file.name,
                    "sha256": result_sha,
                    "bytes": result_file.stat().st_size,
                },
                "metrics": {"AP": 0.42, "AP50": 0.6, "AP75": 0.3},
            },
            "run_id": role.replace("evaluation-", "run-"),
        }
        receipt_path = evaluation_dir / "evaluation-receipt.json"
        _write_json(receipt_path, evaluation_receipt)
        evaluation_paths[role] = receipt_path

    paths: dict[str, Path] = {}
    for name, payload in (
        ("source-receipt.json", source_receipt),
        ("dataset-manifest.json", dataset_manifest),
        ("split-inventory.json", split_inventory),
        ("frozen-image-manifest.json", image_manifest),
        ("model-manifest.json", model_manifest),
    ):
        destination = tmp_path / name
        _write_json(destination, payload)
        paths[name.removesuffix(".json").replace("-", "_")] = destination
    inference_receipt_path = inference_dir / "inference-receipt.json"
    _write_json(inference_receipt_path, inference_receipt)
    paths["inference_receipt"] = inference_receipt_path
    paths["evaluation_a"] = evaluation_paths["evaluation-a"]
    paths["evaluation_b"] = evaluation_paths["evaluation-b"]
    return paths


def _finalize(evidence: dict[str, Path]) -> dict[str, Any]:
    return finalize_bdd100k_detection_benchmark(
        source_receipt=evidence["source_receipt"],
        dataset_manifest=evidence["dataset_manifest"],
        split_inventory=evidence["split_inventory"],
        image_manifest=evidence["frozen_image_manifest"],
        model_manifest=evidence["model_manifest"],
        inference_receipt=evidence["inference_receipt"],
        evaluation_a=evidence["evaluation_a"],
        evaluation_b=evidence["evaluation_b"],
    )


def test_finalizer_accepts_real_prep_freeze_field_shapes_and_redacts_paths(
    tmp_path: Path,
) -> None:
    evidence = _synthetic_evidence(tmp_path, formal=True)
    output = tmp_path / "public" / "benchmark-receipt.json"
    receipt = finalize_bdd100k_detection_benchmark(
        source_receipt=evidence["source_receipt"],
        dataset_manifest=evidence["dataset_manifest"],
        split_inventory=evidence["split_inventory"],
        image_manifest=evidence["frozen_image_manifest"],
        model_manifest=evidence["model_manifest"],
        inference_receipt=evidence["inference_receipt"],
        evaluation_a=evidence["evaluation_a"],
        evaluation_b=evidence["evaluation_b"],
        output=output,
    )

    assert receipt["benchmark_claim_available"] is True
    assert receipt["dataset"]["split_manifest_sha256"]
    assert receipt["evaluator_runs"][0]["metrics"] == {
        "AP": 0.42,
        "AP50": 0.6,
        "AP75": 0.3,
    }
    assert verify_bdd100k_detection_benchmark(output)["report_id"] == receipt["report_id"]

    text = output.read_text(encoding="utf-8")
    assert str(tmp_path) not in text
    assert "predictions.json" not in text
    assert "evaluator-result.json" not in text
    assert "bdd100k.berkeley" not in text


def test_finalizer_rejects_reduced_evidence(tmp_path: Path) -> None:
    """A reduced/community fixture must not cross the formal release gate."""

    evidence = _synthetic_evidence(tmp_path, formal=False)
    with pytest.raises(BDD100KFinalizeError, match="10000|MD5|formal"):
        _finalize(evidence)


def test_finalizer_requires_split_inventory(tmp_path: Path) -> None:
    evidence = _synthetic_evidence(tmp_path, formal=True)
    with pytest.raises(TypeError, match="split_inventory"):
        finalize_bdd100k_detection_benchmark(
            source_receipt=evidence["source_receipt"],
            dataset_manifest=evidence["dataset_manifest"],
            image_manifest=evidence["frozen_image_manifest"],
            model_manifest=evidence["model_manifest"],
            inference_receipt=evidence["inference_receipt"],
            evaluation_a=evidence["evaluation_a"],
            evaluation_b=evidence["evaluation_b"],
        )


def test_finalizer_is_order_invariant_and_requires_identical_evaluator_metrics(
    tmp_path: Path,
) -> None:
    evidence = _synthetic_evidence(tmp_path, formal=True)
    kwargs = {
        "source_receipt": evidence["source_receipt"],
        "dataset_manifest": evidence["dataset_manifest"],
        "split_inventory": evidence["split_inventory"],
        "image_manifest": evidence["frozen_image_manifest"],
        "model_manifest": evidence["model_manifest"],
        "inference_receipt": evidence["inference_receipt"],
    }
    first = finalize_bdd100k_detection_benchmark(
        **kwargs, evaluation_a=evidence["evaluation_a"], evaluation_b=evidence["evaluation_b"]
    )
    reversed_order = finalize_bdd100k_detection_benchmark(
        **kwargs, evaluation_a=evidence["evaluation_b"], evaluation_b=evidence["evaluation_a"]
    )
    assert reversed_order == first

    changed = json.loads(evidence["evaluation_b"].read_text(encoding="utf-8"))
    changed["evaluator"]["metrics"]["AP"] = 0.41
    _write_json(evidence["evaluation_b"], changed)
    with pytest.raises(BDD100KFinalizeError, match="identical metrics"):
        finalize_bdd100k_detection_benchmark(
            **kwargs,
            evaluation_a=evidence["evaluation_a"],
            evaluation_b=evidence["evaluation_b"],
        )


@pytest.mark.parametrize(
    ("tamper", "message"),
    [
        ("prediction", "prediction artifact hash"),
        ("result", "evaluator result artifact hash"),
        ("manifest", "manifest_sha256"),
    ],
)
def test_finalizer_rejects_tampered_frozen_inputs(
    tmp_path: Path, tamper: str, message: str
) -> None:
    evidence = _synthetic_evidence(tmp_path, formal=True)
    if tamper == "prediction":
        prediction = evidence["inference_receipt"].parent / "predictions.json"
        prediction.write_bytes(prediction.read_bytes() + b"tampered")
    elif tamper == "result":
        result = evidence["evaluation_a"].parent / "evaluator-result.json"
        result.write_bytes(result.read_bytes() + b"tampered")
    else:
        frozen_path = evidence["frozen_image_manifest"]
        frozen = load_strict_json(frozen_path)
        assert isinstance(frozen, dict)
        frozen["manifest_sha256"] = _sha_tag("tampered frozen manifest")
        _write_json(frozen_path, frozen)

    with pytest.raises((BDD100KFinalizeError, ValueError), match=message):
        _finalize(evidence)


def test_finalizer_rejects_path_escape_in_evaluator_receipt(tmp_path: Path) -> None:
    for unsafe_path in ("../../outside.json", "C:/outside.json", "..\\outside.json"):
        case_dir = tmp_path / hashlib.sha256(unsafe_path.encode()).hexdigest()[:8]
        case_dir.mkdir()
        evidence = _synthetic_evidence(case_dir, formal=True)
        receipt = json.loads(evidence["evaluation_a"].read_text(encoding="utf-8"))
        receipt["evaluator"]["result"]["path"] = unsafe_path
        _write_json(evidence["evaluation_a"], receipt)
        with pytest.raises(
            BDD100KFinalizeError,
            match="path must be relative|portable|escapes|dot segments",
        ):
            _finalize(evidence)


def test_finalizer_requires_source_archive_and_split_inventory_bindings(tmp_path: Path) -> None:
    missing_archives_dir = tmp_path / "missing-archives"
    missing_archives_dir.mkdir()
    evidence = _synthetic_evidence(missing_archives_dir, formal=True)
    frozen = load_strict_json(evidence["frozen_image_manifest"])
    assert isinstance(frozen, dict)
    frozen.pop("source_archives")
    # Re-freeze so the tampered object remains internally self-consistent; the
    # finalizer must still reject the missing provenance link.
    frozen.pop("frozen", None)
    frozen.pop("freeze_schema_version", None)
    frozen.pop("manifest_sha256", None)
    replacement = freeze_image_manifest(frozen)
    _write_json(evidence["frozen_image_manifest"], replacement)
    with pytest.raises(BDD100KFinalizeError, match="source archives"):
        _finalize(evidence)

    split_mismatch_dir = tmp_path / "split-mismatch"
    split_mismatch_dir.mkdir()
    evidence = _synthetic_evidence(split_mismatch_dir, formal=True)
    split = load_strict_json(evidence["split_inventory"])
    assert isinstance(split, dict)
    split["image_count"] = 999
    _write_json(evidence["split_inventory"], split)
    with pytest.raises(BDD100KFinalizeError, match="split inventory image count"):
        _finalize(evidence)


def _formal_source_attestation() -> tuple[dict[str, Any], dict[str, Any]]:
    source: dict[str, Any] = {
        "schema_version": "roadsense.bdd100k-detection-source-receipt/v1",
        "source_archives": [
            {
                "role": "images_val_zip",
                "format": "zip",
                "official_package_md5": finalizer_module.BDD100K_OFFICIAL_IMAGES_MD5,
                "official_source_url": finalizer_module.BDD100K_OFFICIAL_SOURCE_PAGE,
            },
            {
                "role": "det_20_labels",
                "format": "zip",
                "official_package_md5": finalizer_module.BDD100K_OFFICIAL_LABELS_MD5,
                "official_source_url": finalizer_module.BDD100K_OFFICIAL_SOURCE_PAGE,
            },
        ],
    }
    dataset = {"source_url": finalizer_module.BDD100K_OFFICIAL_SOURCE_PAGE}
    return source, dataset


def test_formal_source_gate_rejects_wrong_md5_or_non_zip_labels() -> None:
    source, dataset = _formal_source_attestation()
    _validate_official_source_attestation(
        source,
        image_count=finalizer_module.BDD100K_OFFICIAL_IMAGE_COUNT,
        dataset_manifest=dataset,
    )

    wrong_md5 = json.loads(json.dumps(source))
    wrong_md5["source_archives"][1]["official_package_md5"] = "e72531b982bbb42efbaaf93223527284"
    with pytest.raises(BDD100KFinalizeError, match="published official MD5"):
        _validate_official_source_attestation(
            wrong_md5,
            image_count=finalizer_module.BDD100K_OFFICIAL_IMAGE_COUNT,
            dataset_manifest=dataset,
        )

    missing_md5 = json.loads(json.dumps(source))
    del missing_md5["source_archives"][0]["official_package_md5"]
    with pytest.raises(BDD100KFinalizeError, match="published official MD5"):
        _validate_official_source_attestation(
            missing_md5,
            image_count=finalizer_module.BDD100K_OFFICIAL_IMAGE_COUNT,
            dataset_manifest=dataset,
        )

    non_zip = json.loads(json.dumps(source))
    non_zip["source_archives"][1]["format"] = "json"
    with pytest.raises(BDD100KFinalizeError, match="official ZIP"):
        _validate_official_source_attestation(
            non_zip,
            image_count=finalizer_module.BDD100K_OFFICIAL_IMAGE_COUNT,
            dataset_manifest=dataset,
        )


def test_formal_evaluator_gate_requires_pycocotools_and_result_file(tmp_path: Path) -> None:
    fixture_dir = tmp_path / "formal-gate-test"
    fixture_dir.mkdir()
    evidence = _synthetic_evidence(fixture_dir, formal=True)
    evaluation = json.loads(evidence["evaluation_a"].read_text(encoding="utf-8"))
    base = evidence["evaluation_a"].parent
    common = {
        "base": base,
        "expected_ground_truth_sha256": _sha_tag("det_val labels"),
        "expected_image_manifest_sha256": evaluation["dataset"]["image_manifest_sha256"],
        "expected_prediction_sha256": evaluation["prediction"]["sha256"],
        "expected_image_count": finalizer_module.BDD100K_OFFICIAL_IMAGE_COUNT,
    }
    evaluation["evaluator"]["packages"].pop("pycocotools", None)

    with pytest.raises(BDD100KFinalizeError, match="validated lock"):
        _validate_evaluator_receipt(evaluation, **common)

    evaluation["evaluator"]["packages"]["pycocotools"] = "2.0.10"
    with pytest.raises(BDD100KFinalizeError, match="validated lock"):
        _validate_evaluator_receipt(evaluation, **common)

    evaluation["evaluator"]["packages"] = dict(finalizer_module.BDD100K_REQUIRED_EVALUATOR_PACKAGES)
    evaluation["evaluator"]["returncode"] = 0
    evaluation["evaluator"]["timed_out"] = False
    evaluation["evaluator"].pop("result_source", None)
    with pytest.raises(BDD100KFinalizeError, match="result file"):
        _validate_evaluator_receipt(evaluation, **common)

    evaluation["evaluator"]["result_source"] = "stdout_fallback"
    with pytest.raises(BDD100KFinalizeError, match="result file"):
        _validate_evaluator_receipt(evaluation, **common)

    evaluation["evaluator"]["result_source"] = "file"
    accepted = _validate_evaluator_receipt(evaluation, **common)
    assert accepted["packages"]["pycocotools"] == "2.0.7"
