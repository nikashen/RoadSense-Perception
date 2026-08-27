from __future__ import annotations

import json
import re
from pathlib import Path

from roadsense import __version__
from roadsense.fixture import build_demo_payload
from roadsense.json_io import canonical_sha256

ROOT = Path(__file__).parents[1]


def test_public_version_is_consistent_across_package_and_demo() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version = "([^"]+)"$', pyproject, flags=re.MULTILINE)
    assert match is not None
    assert match.group(1) == __version__

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    index = (ROOT / "src" / "roadsense" / "web" / "index.html").read_text(encoding="utf-8")
    assert f"`{__version__}` capability preview" in readme
    assert f"v{__version__}" in index
    assert f"app.js?v={__version__}" in index
    assert f"styles.css?v={__version__}" in index


def test_pages_deployment_waits_for_successful_main_push_ci() -> None:
    workflow = (ROOT / ".github" / "workflows" / "pages.yml").read_text(encoding="utf-8")
    assert 'workflows: ["RoadSense CI"]' in workflow
    assert "github.event.workflow_run.conclusion == 'success'" in workflow
    assert "github.event.workflow_run.event == 'push'" in workflow
    assert "github.event.workflow_run.head_branch == 'main'" in workflow
    assert "github.event.workflow_run.head_sha" in workflow
    assert "ROADSENSE_SOURCE_SHA" in workflow


def test_fixture_manifest_hash_matches_generated_payload() -> None:
    """Keep the Pages evidence identity bound to the actual fixture payload."""

    manifest = json.loads((ROOT / "configs" / "fixture_manifest.json").read_text(encoding="utf-8"))
    assert manifest["content_sha256"] == canonical_sha256(build_demo_payload())


def test_sanitized_real_receipt_has_no_local_paths() -> None:
    receipt = json.loads(
        (ROOT / "docs" / "REAL_EVALUATION_RECEIPT.json").read_text(encoding="utf-8")
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
