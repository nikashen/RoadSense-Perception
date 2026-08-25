from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from roadsense.api import create_app
from roadsense.api.models import DemoResponse


def test_api_demo_uses_display_canvas_and_scaled_xywh() -> None:
    with TestClient(create_app()) as client:
        payload = client.get("/api/v1/demo").json()

    assert payload["canvas"] == {"width": 960, "height": 540}
    assert payload["segmentation_categories"] == [
        {"id": 0, "label": "background", "color": "#071411"},
        {"id": 1, "label": "road", "color": "#42e1c3"},
        {"id": 2, "label": "car", "color": "#64d8cb"},
        {"id": 3, "label": "vulnerable road user", "color": "#c8a7ff"},
        {"id": 4, "label": "sidewalk", "color": "#a99aff"},
    ]
    assert len(payload["frames"]) == 24
    first_object = payload["frames"][0]["objects"][0]
    assert first_object["bbox"] == pytest.approx(
        [483.0086206896551, 287.19, 73.5, 41.010000000000005]
    )
    assert payload["evidence"]["evaluation_authorized"] is False
    assert payload["evidence"]["frozen"] is False


def test_api_demo_contract_rejects_frame_count_drift() -> None:
    with TestClient(create_app()) as client:
        payload = client.get("/api/v1/demo").json()
    payload["frames"] = payload["frames"][:-1]
    with pytest.raises(ValidationError, match="exactly 24"):
        DemoResponse.model_validate(payload)


def test_api_demo_contract_rejects_timestamp_cadence_drift() -> None:
    with TestClient(create_app()) as client:
        payload = client.get("/api/v1/demo").json()
    payload["frames"][1]["timestamp_ms"] = 101
    with pytest.raises(ValidationError, match="cadence_ms"):
        DemoResponse.model_validate(payload)


def test_api_demo_contract_rejects_non_fixture_cadence() -> None:
    with TestClient(create_app()) as client:
        payload = client.get("/api/v1/demo").json()
    payload["cadence_ms"] = 200
    payload["fps"] = 5
    for index, frame in enumerate(payload["frames"]):
        frame["timestamp_ms"] = index * 200
    with pytest.raises(ValidationError, match="cadence_ms"):
        DemoResponse.model_validate(payload)


def test_pages_build_contains_relative_fixture_contract() -> None:
    from scripts.build_pages import build

    output = Path("dist") / ".pytest-pages-contract"
    try:
        built = build(output)
        manifest = json.loads((built / "manifest.json").read_text(encoding="utf-8"))
        app_source = (built / "app.js").read_text(encoding="utf-8")

        assert manifest["runtime"] == "deterministic_geometric_fixture"
        assert manifest["benchmark_claim_available"] is False
        assert manifest["fixture"] == {
            "fixture_id": "roadsense-city-loop-v2",
            "frame_count": 24,
            "canvas": {"width": 960, "height": 540},
            "cadence_ms": 100,
            "payload_sha256": "5358bb72f4a375246beb45986978a050b28976cd9218f472bbb6fcacfeb552f1",
        }
        demo = json.loads((built / "demo.json").read_text(encoding="utf-8"))
        assert demo["fixture_id"] == "roadsense-city-loop-v2"
        assert len(demo["frames"]) == 24
        assert demo["evidence"]["benchmark_claim_available"] is False
        assert "Array.from({ length: 24 }" in app_source
        assert "api_unavailable:" in app_source
        assert "dataset.label" in app_source
        assert all((built / asset).is_file() for asset in manifest["assets"])
        assert set(manifest["asset_sha256"]) == set(manifest["assets"])
    finally:
        shutil.rmtree(output, ignore_errors=True)


def test_pages_builder_rejects_destructive_output_paths() -> None:
    from scripts.build_pages import build

    for unsafe in (Path("."), Path("src"), Path(".git"), Path("..")):
        with pytest.raises(ValueError, match="child of repository dist"):
            build(unsafe)


def test_pages_verifier_accepts_built_artifact() -> None:
    from scripts.build_pages import build
    from scripts.verify_pages import verify

    output = Path("dist") / ".pytest-pages-verifier"
    try:
        build(output)
        result = verify(output)
        assert result["status"] == "ok"
        assert result["frame_count"] == 24
    finally:
        shutil.rmtree(output, ignore_errors=True)


def test_pages_verifier_rejects_unlisted_artifacts() -> None:
    from scripts.build_pages import build
    from scripts.verify_pages import verify

    output = Path("dist") / ".pytest-pages-extra"
    try:
        build(output)
        (output / "unexpected.txt").write_text("not in manifest", encoding="utf-8")
        with pytest.raises(ValueError, match="unexpected files"):
            verify(output)
    finally:
        shutil.rmtree(output, ignore_errors=True)
