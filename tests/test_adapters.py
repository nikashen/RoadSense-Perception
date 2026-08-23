from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from roadsense.adapters import (
    MODEL_ARTIFACT_SCHEMA,
    AdapterRegistry,
    ArtifactVerificationError,
    ModelArtifactManifest,
    load_artifact_manifest,
    verify_artifact_manifest,
)
from roadsense.cli import main
from roadsense.contracts import TaskKind
from roadsense.json_io import canonical_sha256


def _manifest_payload(
    artifact_hash: str,
    *,
    artifact_path: str = "weights/model.onnx",
    artifact_size_bytes: int | None = None,
) -> dict[str, object]:
    return {
        "schema_version": MODEL_ARTIFACT_SCHEMA,
        "artifact_id": "demo-detector-v1",
        "artifact_path": artifact_path,
        "artifact_sha256": artifact_hash,
        "artifact_size_bytes": artifact_size_bytes,
        "artifact_format": "onnx",
        "framework": "custom",
        "framework_version": "1.0",
        "backend": "onnxruntime",
        "backend_version": "1.18.0",
        "tasks": ["detection", "tracking"],
        "ontology": [
            {"category_id": 0, "label": "background"},
            {"category_id": 1, "label": "car"},
        ],
        "input": {
            "width": 640,
            "height": 360,
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
            "output_shape": [None, 7],
        },
        "preprocessing": {
            "resize_mode": "letterbox",
            "scale": 0.00392156862745098,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "pad_value": 0.0,
        },
        "runtime": {
            "device": "cpu",
            "precision": "fp32",
            "runtime_name": "onnxruntime",
            "runtime_version": "1.18.0",
            "opset": 17,
            "deterministic": True,
        },
        "license_id": "local-only",
        "source": "operator://local-checkpoint",
        "notes": "test artifact; no model is loaded by the contract tests",
    }


def test_manifest_binds_ontology_preprocessing_and_runtime() -> None:
    payload = _manifest_payload("a" * 64)
    manifest = ModelArtifactManifest.model_validate(payload)
    assert manifest.schema_version == MODEL_ARTIFACT_SCHEMA
    assert manifest.tasks == (TaskKind.DETECTION, TaskKind.TRACKING)
    assert [item.category_id for item in manifest.ontology] == [0, 1]
    assert manifest.preprocessing.resize_mode == "letterbox"
    assert manifest.input.coordinate_space == "original_xyxy"
    assert manifest.output.output_shape == (None, 7)


def test_manifest_freezes_and_rejects_non_finite_graph_metadata() -> None:
    payload = _manifest_payload("a" * 64)
    payload["graph_metadata"] = {"opset": 17, "dynamic_axes": {"boxes": [None, 7]}}
    manifest = ModelArtifactManifest.model_validate(payload)
    with pytest.raises(TypeError, match="immutable"):
        manifest.graph_metadata["opset"] = 18
    payload["graph_metadata"] = {"bad": float("nan")}
    with pytest.raises(ValidationError, match="finite"):
        ModelArtifactManifest.model_validate(payload)


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("artifact_path", "../escape.onnx", "relative"),
        ("artifact_path", "C:/outside.onnx", "relative"),
        ("artifact_path", " weights/model.onnx", "whitespace"),
        ("artifact_path", "weights//model.onnx", "normalized"),
        ("artifact_sha256", "0" * 64, "zero"),
        ("tasks", ["detection", "detection"], "unique"),
        (
            "ontology",
            [{"category_id": 1, "label": "car"}, {"category_id": 1, "label": "truck"}],
            "category IDs",
        ),
        ("artifact_size_bytes", True, "integer"),
        (
            "ontology",
            [{"category_id": 0, "label": " background"}],
            "surrounding whitespace",
        ),
    ],
)
def test_manifest_rejects_unsafe_or_ambiguous_values(field: str, value: object, match: str) -> None:
    payload = _manifest_payload("a" * 64)
    payload[field] = value
    with pytest.raises(ValidationError, match=match):
        ModelArtifactManifest.model_validate(payload)


def test_verify_artifact_hash_and_size(tmp_path: Path) -> None:
    artifact = tmp_path / "weights" / "model.onnx"
    artifact.parent.mkdir()
    artifact.write_bytes(b"deterministic-test-artifact")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    manifest = ModelArtifactManifest.model_validate(
        _manifest_payload(digest, artifact_size_bytes=artifact.stat().st_size)
    )
    receipt = verify_artifact_manifest(manifest, artifact_root=tmp_path)
    assert receipt.verified is True
    assert receipt.artifact_sha256 == digest
    assert receipt.artifact_size_bytes == artifact.stat().st_size
    assert receipt.manifest_sha256 == canonical_sha256(manifest.model_dump(mode="json"))

    artifact.write_bytes(b"tampered")
    with pytest.raises(ArtifactVerificationError, match="mismatch"):
        verify_artifact_manifest(manifest, artifact_root=tmp_path)


def test_verify_rejects_symlink_escape(tmp_path: Path) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside.bin"
    outside.write_bytes(b"outside-artifact")
    link = tmp_path / "weights" / "model.onnx"
    link.parent.mkdir()
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")
    digest = hashlib.sha256(outside.read_bytes()).hexdigest()
    manifest = ModelArtifactManifest.model_validate(_manifest_payload(digest))
    with pytest.raises(ArtifactVerificationError, match="symlink"):
        verify_artifact_manifest(manifest, artifact_root=tmp_path)


def test_registry_requires_explicit_verification(tmp_path: Path) -> None:
    artifact = tmp_path / "weights" / "model.onnx"
    artifact.parent.mkdir()
    artifact.write_bytes(b"registry-artifact")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    manifest = ModelArtifactManifest.model_validate(
        _manifest_payload(digest, artifact_size_bytes=artifact.stat().st_size)
    )

    class Adapter:
        adapter_id = manifest.artifact_id

        def __init__(self) -> None:
            self.manifest = manifest

        def infer(self, frame: object) -> object:
            return frame

    registry = AdapterRegistry()
    registry.register(Adapter())
    assert not registry.is_verified(manifest.artifact_id)
    with pytest.raises(ArtifactVerificationError, match="no verified"):
        registry.require_verified(manifest.artifact_id)

    receipt = registry.register(Adapter(), artifact_root=tmp_path, replace=True)
    assert receipt is not None
    assert registry.is_verified(manifest.artifact_id)
    assert registry.for_task("detection")
    assert registry.require_verified(manifest.artifact_id).infer("frame") == "frame"


def test_cli_manifest_only_dry_run_does_not_load_model(tmp_path: Path, capsys) -> None:
    artifact = b"cli-dry-run-artifact"
    digest = hashlib.sha256(artifact).hexdigest()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(_manifest_payload(digest), ensure_ascii=False), encoding="utf-8"
    )
    assert main(["audit-artifact", str(manifest_path), "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "manifest_valid"
    assert result["verification_scope"] == "manifest_only"
    assert result["verified"] is False
    assert result["model_loaded"] is False


def test_cli_artifact_verification_writes_receipt(tmp_path: Path, capsys) -> None:
    artifact = tmp_path / "weights" / "model.onnx"
    artifact.parent.mkdir()
    artifact.write_bytes(b"cli-verified-artifact")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            _manifest_payload(digest, artifact_size_bytes=artifact.stat().st_size),
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    output = tmp_path / "receipt.json"
    assert (
        main(
            [
                "verify-artifact",
                str(manifest_path),
                "--root",
                str(tmp_path),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert str(output.resolve()) in capsys.readouterr().out
    receipt = json.loads(output.read_text(encoding="utf-8"))
    assert receipt["status"] == "verified"
    assert receipt["model_loaded"] is False
    assert receipt["verified"] is True


def test_load_artifact_manifest_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text('{"schema_version":"x","schema_version":"y"}', encoding="utf-8")
    with pytest.raises(ArtifactVerificationError, match="load artifact manifest"):
        load_artifact_manifest(path)
