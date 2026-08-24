from __future__ import annotations

import json
import math
from copy import deepcopy
from pathlib import Path

import pytest
from pydantic import ValidationError

from roadsense.bdd_benchmark import (
    BDD100K_DETECTION_BENCHMARK_SCHEMA,
    BDD100K_DETECTION_SCOPE,
    BDD100K_DEVKIT_COMMIT,
    BDD100K_DEVKIT_ID,
    BDD100K_DEVKIT_REPOSITORY,
    BDD100KBenchmarkReceiptError,
    BDD100KDetectionBenchmarkReceipt,
    build_bdd100k_detection_receipt,
    compute_bdd100k_detection_report_id,
    load_bdd100k_detection_receipt,
    validate_bdd100k_detection_receipt,
)


def _sha256(character: str) -> str:
    return character * 64


def _payload() -> dict[str, object]:
    """Synthetic evidence only; no BDD100K file or model is needed here."""

    return {
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
            "archive_sha256": _sha256("1"),
            "tree_sha256": _sha256("2"),
            "ground_truth_sha256": _sha256("3"),
            "split_manifest_sha256": _sha256("4"),
        },
        "model": {
            "model_id": "yolox-s-bdd100k-v1",
            "artifact_sha256": _sha256("5"),
            "manifest_sha256": _sha256("6"),
            "ontology_map_sha256": _sha256("7"),
        },
        "inference": {
            "config_sha256": _sha256("8"),
            "prediction_sha256": _sha256("9"),
        },
        "evaluator": {
            "evaluator_id": BDD100K_DEVKIT_ID,
            "repository": BDD100K_DEVKIT_REPOSITORY,
            "commit": BDD100K_DEVKIT_COMMIT,
            "config_sha256": _sha256("a"),
            "dependencies": {
                "lock_sha256": _sha256("b"),
                "packages": {
                    "numpy": "1.26.4",
                    "pydantic": "2.8.2",
                },
            },
        },
        "evaluator_runs": [
            {
                "role": "independent_a",
                "run_id": "devkit-run-a",
                "output_sha256": _sha256("c"),
                "metrics": {"mAP": 0.384, "AP_car": 0.612},
            },
            {
                "role": "independent_b",
                "run_id": "devkit-run-b",
                "output_sha256": _sha256("d"),
                "metrics": {"AP_car": 0.612, "mAP": 0.384},
            },
        ],
    }


def _complete_payload() -> dict[str, object]:
    return build_bdd100k_detection_receipt(_payload()).model_dump(mode="json")


def _rebind(payload: dict[str, object]) -> dict[str, object]:
    payload["report_id"] = compute_bdd100k_detection_report_id(payload)
    return payload


def test_builds_a_full_canonical_receipt_without_real_inputs() -> None:
    receipt = build_bdd100k_detection_receipt(_payload())
    payload = receipt.model_dump(mode="json")

    assert isinstance(receipt, BDD100KDetectionBenchmarkReceipt)
    assert payload["schema_version"] == BDD100K_DETECTION_BENCHMARK_SCHEMA
    assert payload["scope"] == BDD100K_DETECTION_SCOPE
    assert payload["benchmark_claim_available"] is True
    assert payload["dataset"]["split"] == "val"  # type: ignore[index]
    assert payload["evaluator"]["commit"] == BDD100K_DEVKIT_COMMIT  # type: ignore[index]
    assert payload["report_id"] == compute_bdd100k_detection_report_id(payload)
    assert len(payload["report_id"]) == 64  # type: ignore[arg-type]
    assert validate_bdd100k_detection_receipt(payload) == receipt


def test_report_id_is_canonical_and_binds_every_receipt_field() -> None:
    first = build_bdd100k_detection_receipt(_payload()).model_dump(mode="json")
    reordered = _payload()
    evaluator = reordered["evaluator"]
    assert isinstance(evaluator, dict)
    dependencies = evaluator["dependencies"]
    assert isinstance(dependencies, dict)
    packages = dependencies["packages"]
    assert isinstance(packages, dict)
    dependencies["packages"] = {"pydantic": packages["pydantic"], "numpy": packages["numpy"]}
    second = build_bdd100k_detection_receipt(reordered).model_dump(mode="json")

    assert first["report_id"] == second["report_id"]

    tampered = deepcopy(first)
    inference = tampered["inference"]
    assert isinstance(inference, dict)
    inference["prediction_sha256"] = _sha256("e")
    with pytest.raises(ValidationError, match="report_id"):
        validate_bdd100k_detection_receipt(tampered)

    with pytest.raises(BDD100KBenchmarkReceiptError, match="report_id"):
        build_bdd100k_detection_receipt(tampered)


@pytest.mark.parametrize(
    ("path", "value", "match"),
    [
        (("scope",), "another dataset", "scope"),
        (("benchmark_claim_available",), False, "benchmark_claim_available"),
        (("benchmark_claim_available",), "true", "boolean"),
        (("dataset", "task"), "segmentation", "task"),
        (("dataset", "split"), "test", "split"),
        (("dataset", "release"), "2023", "release"),
        (("evaluator", "evaluator_id"), "local-evaluator", "evaluator_id"),
        (("evaluator", "repository"), "someone/fork", "repository"),
        (("evaluator", "commit"), _sha256("a"), "commit"),
    ],
)
def test_rejects_out_of_scope_claims(path: tuple[str, ...], value: object, match: str) -> None:
    payload = _complete_payload()
    target: dict[str, object] = payload
    for key in path[:-1]:
        nested = target[key]
        assert isinstance(nested, dict)
        target = nested
    target[path[-1]] = value
    _rebind(payload)

    with pytest.raises(ValidationError, match=match):
        validate_bdd100k_detection_receipt(payload)


@pytest.mark.parametrize(
    ("path", "value", "match"),
    [
        (("dataset", "archive_sha256"), "f" * 63, "SHA-256"),
        (("dataset", "tree_sha256"), "0" * 64, "all-zero"),
        (("model", "artifact_sha256"), "A" * 64, "SHA-256"),
        (("inference", "prediction_sha256"), "not-a-hash", "SHA-256"),
        (("evaluator", "config_sha256"), "0" * 64, "all-zero"),
        (("evaluator", "dependencies", "lock_sha256"), "f" * 63, "SHA-256"),
        (("evaluator_runs", "0", "output_sha256"), "0" * 64, "all-zero"),
    ],
)
def test_rejects_wrong_or_placeholder_hashes(
    path: tuple[str, ...], value: object, match: str
) -> None:
    payload = _complete_payload()
    target: object = payload
    for key in path[:-1]:
        if isinstance(target, list):
            target = target[int(key)]
        else:
            assert isinstance(target, dict)
            target = target[key]
    assert isinstance(target, dict)
    target[path[-1]] = value
    _rebind(payload)

    with pytest.raises(ValidationError, match=match):
        validate_bdd100k_detection_receipt(payload)


def test_rejects_path_leakage_and_unsafe_dependency_versions() -> None:
    payload = _complete_payload()
    model = payload["model"]
    assert isinstance(model, dict)
    model["model_id"] = r"C:\Users\operator\weights\model.onnx"
    _rebind(payload)
    with pytest.raises(ValidationError, match="paths"):
        validate_bdd100k_detection_receipt(payload)

    payload = _complete_payload()
    evaluator = payload["evaluator"]
    assert isinstance(evaluator, dict)
    dependencies = evaluator["dependencies"]
    assert isinstance(dependencies, dict)
    packages = dependencies["packages"]
    assert isinstance(packages, dict)
    packages["torch"] = "/home/operator/.cache/torch"
    _rebind(payload)
    with pytest.raises(ValidationError, match="paths"):
        validate_bdd100k_detection_receipt(payload)


@pytest.mark.parametrize("bad_metric", [math.nan, math.inf, -math.inf, True, "0.384"])
def test_rejects_non_finite_or_lossy_metric_values(bad_metric: object) -> None:
    payload = _complete_payload()
    runs = payload["evaluator_runs"]
    assert isinstance(runs, list)
    first = runs[0]
    assert isinstance(first, dict)
    metrics = first["metrics"]
    assert isinstance(metrics, dict)
    metrics["mAP"] = bad_metric

    with pytest.raises(ValidationError, match="finite|number"):
        validate_bdd100k_detection_receipt(payload)


def test_requires_two_ordered_reproducible_evaluator_runs() -> None:
    payload = _complete_payload()
    runs = payload["evaluator_runs"]
    assert isinstance(runs, list)
    payload["evaluator_runs"] = runs[:1]
    _rebind(payload)
    with pytest.raises(ValidationError, match="2"):
        validate_bdd100k_detection_receipt(payload)

    payload = _complete_payload()
    runs = payload["evaluator_runs"]
    assert isinstance(runs, list)
    first, second = runs
    assert isinstance(first, dict) and isinstance(second, dict)
    second["run_id"] = first["run_id"]
    _rebind(payload)
    with pytest.raises(ValidationError, match="distinct run_id"):
        validate_bdd100k_detection_receipt(payload)

    payload = _complete_payload()
    runs = payload["evaluator_runs"]
    assert isinstance(runs, list)
    payload["evaluator_runs"] = list(reversed(runs))
    _rebind(payload)
    with pytest.raises(ValidationError, match="canonically ordered"):
        validate_bdd100k_detection_receipt(payload)

    payload = _complete_payload()
    runs = payload["evaluator_runs"]
    assert isinstance(runs, list)
    second = runs[1]
    assert isinstance(second, dict)
    metrics = second["metrics"]
    assert isinstance(metrics, dict)
    metrics["mAP"] = 0.385
    _rebind(payload)
    with pytest.raises(ValidationError, match="identical metrics"):
        validate_bdd100k_detection_receipt(payload)


def test_rejects_unknown_fields_and_mutable_nested_provenance() -> None:
    payload = _complete_payload()
    payload["local_path"] = r"F:\private\benchmark"
    _rebind(payload)
    with pytest.raises(ValidationError, match="Extra inputs"):
        validate_bdd100k_detection_receipt(payload)

    receipt = build_bdd100k_detection_receipt(_payload())
    with pytest.raises(TypeError, match="immutable"):
        receipt.evaluator.dependencies.packages["numpy"] = "2.0.0"
    with pytest.raises(TypeError, match="immutable"):
        receipt.evaluator_runs[0].metrics["mAP"] = 0.0


def test_load_uses_strict_json_for_duplicate_keys_and_non_finite_constants(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema_version":"one","schema_version":"two"}', encoding="utf-8")
    with pytest.raises(BDD100KBenchmarkReceiptError, match="unable to load"):
        load_bdd100k_detection_receipt(duplicate)

    non_finite = tmp_path / "non-finite.json"
    non_finite.write_text('{"metric": NaN}', encoding="utf-8")
    with pytest.raises(BDD100KBenchmarkReceiptError, match="unable to load"):
        load_bdd100k_detection_receipt(non_finite)


def test_load_round_trips_a_complete_receipt(tmp_path: Path) -> None:
    path = tmp_path / "bdd100k-receipt.json"
    path.write_text(json.dumps(_complete_payload(), sort_keys=True), encoding="utf-8")

    receipt = load_bdd100k_detection_receipt(path)
    assert receipt.report_id == compute_bdd100k_detection_report_id(receipt.model_dump(mode="json"))
