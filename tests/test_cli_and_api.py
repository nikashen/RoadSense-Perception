from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from roadsense.api import create_app
from roadsense.api.models import DemoResponse, ReportResponse
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


def test_cli_report_returns_error_for_unwritable_output(tmp_path, capsys) -> None:
    output_dir = tmp_path / "already-a-directory"
    output_dir.mkdir()
    assert main(["report", "--output", str(output_dir)]) == 2
    assert "report generation failed" in capsys.readouterr().err


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
        assert readiness["service_mode"] == "fixture_replay"
        assert readiness["verification_level"] == "fixture"
        assert readiness["model_loaded"] is False
        demo = client.get("/api/v1/demo")
        assert demo.status_code == 200
        assert len(demo.json()["frames"]) == 24
        report = client.get("/api/v1/report").json()
        assert report["evaluation_authorized"] is False
        validated_report = ReportResponse.model_validate(report)
        with pytest.raises(TypeError, match="immutable"):
            validated_report.details["tampered"] = True
        report["details"] = {"tampered": True}
        with pytest.raises(ValidationError, match="report_id"):
            ReportResponse.model_validate(report)


def test_api_report_rejects_inconsistent_evidence_flags() -> None:
    from roadsense.evidence import build_fixture_report

    payload = build_fixture_report()
    payload["evaluation_authorized"] = True
    with pytest.raises(ValidationError, match="fixture reports"):
        ReportResponse.model_validate(payload)

    payload = build_fixture_report()
    payload["dataset_manifest_sha256"] = "not-a-sha256"
    with pytest.raises(ValidationError, match="sha256"):
        ReportResponse.model_validate(payload)


@pytest.mark.parametrize("value", ["0.5", True, None])
def test_api_report_rejects_lossy_metric_coercion(value: object) -> None:
    from roadsense.evidence import build_fixture_report

    payload = build_fixture_report()
    payload["metrics"] = {"detection_ap50": value}  # type: ignore[dict-item]
    with pytest.raises(ValidationError, match="metrics"):
        ReportResponse.model_validate(payload)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("image_size", "width"), "640"),
        (("frames", 0, "frame_index"), "0"),
        (("frames", 0, "objects", 0, "confidence"), "0.9"),
        (("frames", 0, "objects", 0, "track_id"), True),
        (("fps",), "10"),
    ],
)
def test_api_response_rejects_numeric_coercion(path: tuple[object, ...], value: object) -> None:
    payload = create_app()
    with TestClient(payload) as client:
        demo = client.get("/api/v1/demo").json()
    target: object = demo
    for key in path[:-1]:
        target = target[key]  # type: ignore[index]
    target[path[-1]] = value  # type: ignore[index]
    with pytest.raises(ValidationError):
        DemoResponse.model_validate(demo)
