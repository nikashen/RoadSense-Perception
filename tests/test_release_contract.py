from __future__ import annotations

import json
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
