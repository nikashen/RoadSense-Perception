from __future__ import annotations

import json
import re
from pathlib import Path

from roadsense.fixture import build_demo_payload
from roadsense.json_io import canonical_sha256


def test_fixture_manifest_hash_matches_generated_payload() -> None:
    """Keep the Pages evidence identity bound to the actual fixture payload."""

    manifest = json.loads(
        (Path(__file__).parents[1] / "configs" / "fixture_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["content_sha256"] == canonical_sha256(build_demo_payload())


def test_sanitized_real_receipt_has_no_local_paths() -> None:
    receipt = json.loads(
        (Path(__file__).parents[1] / "docs" / "REAL_EVALUATION_RECEIPT.json").read_text(
            encoding="utf-8"
        )
    )
    assert receipt["schema_version"] == "roadsense.sanitized-real-evaluation/v1"
    assert receipt["receipt_scope"] == "aggregate_provenance_only"
    assert receipt["evidence"] == {
        "evidence_level": "development",
        "evaluation_authorized": False,
        "frozen": False,
        "benchmark_claim_available": False,
    }
    assert re.fullmatch(r"[0-9a-f]{16}", receipt["report_id"])
    for digest in (
        receipt["dataset"]["archive_sha256"],
        receipt["dataset"]["tree_sha256"],
        receipt["dataset"]["manifest_sha256"],
        receipt["dataset"]["spec_sha256"],
        receipt["model"]["artifact_sha256"],
        receipt["model"]["manifest_sha256"],
        receipt["model"]["dependency_lock_sha256"],
    ):
        assert re.fullmatch(r"[0-9a-f]{64}", digest)
    serialized = json.dumps(receipt, ensure_ascii=False)
    assert "F:\\" not in serialized
    assert "data/raw" not in serialized
    assert "ground-truth.json" not in serialized
    assert "predictions.json" not in serialized
