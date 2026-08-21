from __future__ import annotations

import json

from fastapi.testclient import TestClient

from roadsense.api import create_app
from roadsense.cli import main


def test_cli_smoke_contract(capsys) -> None:
    assert main(["smoke", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["frames"] == 24
    assert payload["benchmark_claim_available"] is False


def test_cli_report_writes_evidence(tmp_path, capsys) -> None:
    output = tmp_path / "fixture.json"
    assert main(["report", "--output", str(output)]) == 0
    assert output.is_file()
    assert json.loads(output.read_text(encoding="utf-8"))["evidence_level"] == "fixture"
    assert str(output.resolve()) in capsys.readouterr().out


def test_cli_audits_fixture_manifest(capsys) -> None:
    assert main(["audit-manifest", "configs/fixture_manifest.json"]) == 0
    assert '"evaluation_authorized": false' in capsys.readouterr().out


def test_api_health_readiness_demo_and_report() -> None:
    with TestClient(create_app()) as client:
        page = client.get("/")
        assert page.status_code == 200
        for asset in ("/app.js", "/styles.css", "/favicon.svg"):
            assert client.get(asset).status_code == 200
        assert client.get("/api/v1/health").json()["status"] == "ok"
        readiness = client.get("/api/v1/readiness").json()
        assert readiness["verification_level"] == "fixture"
        assert readiness["model_loaded"] is False
        demo = client.get("/api/v1/demo")
        assert demo.status_code == 200
        assert len(demo.json()["frames"]) == 24
        report = client.get("/api/v1/report").json()
        assert report["evaluation_authorized"] is False
