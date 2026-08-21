from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from roadsense.api import create_app


def test_api_demo_uses_display_canvas_and_scaled_xywh() -> None:
    with TestClient(create_app()) as client:
        payload = client.get("/api/v1/demo").json()

    assert payload["canvas"] == {"width": 960, "height": 540}
    assert len(payload["frames"]) == 24
    first_object = payload["frames"][0]["objects"][0]
    assert first_object["bbox"] == pytest.approx([100.05, 340.95, 157.5, 86.25])
    assert payload["evidence"]["evaluation_authorized"] is False
    assert payload["evidence"]["frozen"] is False


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
            "fixture_id": "roadsense-city-loop-v1",
            "frame_count": 24,
            "canvas": {"width": 960, "height": 540},
            "cadence_ms": 100,
            "payload_sha256": "5eb6923e345e4c199ac01625449ce227b96cb98fba8876f5e19d4cfbce9ff253",
        }
        assert "Array.from({ length: 24 }" in app_source
        assert all((built / asset).is_file() for asset in manifest["assets"])
        assert set(manifest["asset_sha256"]) == set(manifest["assets"])
    finally:
        shutil.rmtree(output, ignore_errors=True)


def test_pages_builder_rejects_destructive_output_paths() -> None:
    from scripts.build_pages import build

    for unsafe in (Path("."), Path("src"), Path(".git"), Path("..")):
        with pytest.raises(ValueError, match="child of repository dist"):
            build(unsafe)
