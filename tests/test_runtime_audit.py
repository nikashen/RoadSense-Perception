from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from roadsense import __version__
from roadsense.cli import main
from roadsense.json_io import canonical_sha256
from roadsense.runtime import (
    RuntimeAuditRecord,
    RuntimeStage,
    build_fixture_runtime_record,
    compute_runtime_record_id,
)


def test_fixture_runtime_record_is_explicitly_non_benchmark() -> None:
    record = build_fixture_runtime_record()
    payload = record.model_dump(mode="json")

    assert payload["schema_version"] == "roadsense.runtime-audit/v1"
    assert payload["run_mode"] == "fixture_dry_run"
    assert payload["evidence_level"] == "fixture"
    assert payload["benchmark_claim_available"] is False
    assert payload["input_sha256"] == payload["input"]["payload_sha256"]
    assert payload["output_sha256"] == payload["output"]["payload_sha256"]
    assert payload["record_id"] == compute_runtime_record_id(payload)
    assert payload["device"]["python_version"]
    assert payload["dependencies"]
    assert payload["dependencies"]["roadsense-perception"] == __version__
    assert payload["throughput_fps"] is not None
    assert payload["input_sha256"] != canonical_sha256(
        {
            "source": "deterministic_geometric_fixture",
            "fixture_id": payload["input"]["fixture_id"],
            "frame_count": payload["input"]["frame_count"],
            "iterations": payload["input"]["iterations"],
        }
    )
    assert {stage["name"] for stage in payload["stages"]} >= {
        "fixture_generation",
        "evaluation",
        "serialization",
        "inference",
        "rendering",
    }
    assert (
        next(stage for stage in payload["stages"] if stage["name"] == "inference")["measured"]
        is False
    )


def test_fixture_runtime_iterations_bind_input_and_throughput() -> None:
    record = build_fixture_runtime_record(iterations=2)
    assert record.iterations == 2
    assert record.input.iterations == 2
    assert record.input.frame_count == 24
    assert record.throughput_fps is not None
    assert record.throughput_fps > 0


def test_runtime_record_rejects_tampered_hash_and_flags() -> None:
    payload = build_fixture_runtime_record().model_dump(mode="json")
    payload["output_sha256"] = "a" * 64
    with pytest.raises(ValidationError, match="output_sha256"):
        RuntimeAuditRecord.model_validate(payload)

    payload = build_fixture_runtime_record().model_dump(mode="json")
    payload["input"]["source"] = "local_sequence"
    payload["record_id"] = compute_runtime_record_id(payload)
    with pytest.raises(ValidationError, match="deterministic fixture source"):
        RuntimeAuditRecord.model_validate(payload)

    payload = build_fixture_runtime_record().model_dump(mode="json")
    payload["evaluation_authorized"] = True
    payload["record_id"] = compute_runtime_record_id(payload)
    with pytest.raises(ValidationError, match="fixture runtime records"):
        RuntimeAuditRecord.model_validate(payload)


def test_runtime_stage_does_not_allow_fake_unmeasured_timing() -> None:
    with pytest.raises(ValidationError, match="unmeasured stage"):
        RuntimeStage(name="inference", measured=False, duration_ms=0.0, note="missing")
    with pytest.raises(ValidationError, match="throughput"):
        RuntimeStage(
            name="evaluation",
            measured=True,
            duration_ms=10.0,
            items=10,
            throughput_per_s=1.0,
        )


def test_runtime_schema_rejects_numeric_string_coercion() -> None:
    payload = build_fixture_runtime_record().model_dump(mode="json")
    payload["iterations"] = "1"
    with pytest.raises(ValidationError, match="integer"):
        RuntimeAuditRecord.model_validate(payload)

    payload = build_fixture_runtime_record().model_dump(mode="json")
    payload["stages"][0]["duration_ms"] = "1.0"
    payload["record_id"] = compute_runtime_record_id(payload)
    with pytest.raises(ValidationError, match="number"):
        RuntimeAuditRecord.model_validate(payload)


def test_runtime_record_requires_utc_timestamp() -> None:
    payload = build_fixture_runtime_record().model_dump(mode="json")
    payload["started_at_utc"] = datetime.now(timezone.utc).replace(tzinfo=None).isoformat()
    payload["record_id"] = compute_runtime_record_id(payload)
    with pytest.raises(ValidationError, match="UTC"):
        RuntimeAuditRecord.model_validate(payload)


def test_cli_benchmark_writes_json_record(tmp_path, capsys) -> None:
    output = tmp_path / "runtime.json"
    assert main(["benchmark", "--output", str(output), "--iterations", "2"]) == 0
    assert str(output.resolve()) in capsys.readouterr().out
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["iterations"] == 2
    assert payload["benchmark_claim_available"] is False


def test_cli_runtime_audit_json_alias(tmp_path, capsys) -> None:
    output = tmp_path / "runtime-alias.json"
    assert main(["runtime-audit", "--output", str(output), "--json"]) == 0
    stdout = capsys.readouterr().out
    assert json.loads(stdout)["schema_version"] == "roadsense.runtime-audit/v1"
    assert output.is_file()
