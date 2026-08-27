"""Build a relative-path GitHub Pages artifact for the deterministic fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")


def _source_revision(root: Path) -> tuple[str, str]:
    expected_sha = os.getenv("ROADSENSE_SOURCE_SHA", "").lower()
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        commit = completed.stdout.strip().lower()
    except (OSError, subprocess.SubprocessError):
        if _GIT_SHA.fullmatch(expected_sha):
            return expected_sha, "unknown"
        return "unknown", "unknown"
    if not _GIT_SHA.fullmatch(commit):
        return "unknown", "unknown"
    if _GIT_SHA.fullmatch(expected_sha) and commit != expected_sha:
        raise RuntimeError("checked-out source commit does not match ROADSENSE_SOURCE_SHA")
    try:
        status = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=no"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return commit, "unknown"
    return commit, "dirty" if status.stdout.strip() else "clean"


def build(output: Path) -> Path:
    root = Path(__file__).resolve().parents[1]
    source = root / "src" / "roadsense" / "web"
    static_assets = ("index.html", "app.js", "styles.css", "favicon.svg")
    if any(not (source / name).is_file() for name in static_assets):
        raise FileNotFoundError("web assets are incomplete")
    output = output.resolve()
    pages_root = (root / "dist").resolve()
    if output == pages_root or pages_root not in output.parents:
        raise ValueError("Pages output must be a child of repository dist/")

    # Build and validate the generated payload before touching an existing
    # artifact. A missing dependency or stale manifest must not destroy the
    # last known-good Pages directory.
    source_root = str(root / "src")
    if source_root not in sys.path:
        sys.path.insert(0, source_root)
    try:
        from roadsense import __version__
        from roadsense.fixture import build_demo_payload
        from roadsense.json_io import canonical_sha256
    except ImportError as exc:
        raise RuntimeError("building Pages requires the project runtime dependencies") from exc

    payload = build_demo_payload()
    payload_hash = canonical_sha256(payload)
    source_commit, source_tree_state = _source_revision(root)
    fixture_manifest = json.loads(
        (root / "configs" / "fixture_manifest.json").read_text(encoding="utf-8")
    )
    if fixture_manifest.get("content_sha256") != payload_hash:
        raise ValueError("fixture manifest hash does not match the generated demo payload")

    if output.exists():
        try:
            shutil.rmtree(output)
        except OSError as exc:
            raise RuntimeError(
                "Pages output cannot be replaced; stop the local static server or choose another dist/ path"
            ) from exc
    output.mkdir(parents=True)
    for name in static_assets:
        shutil.copy2(source / name, output / name)
    # Generate a Pages replay from the exact same payload served by the local
    # API. This prevents the public artifact from drifting into a second,
    # undocumented fixture schema.
    (output / "demo.json").write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / ".nojekyll").write_text("", encoding="ascii")
    assets = list(static_assets) + ["demo.json", ".nojekyll"]
    asset_hashes = {
        name: hashlib.sha256((output / name).read_bytes()).hexdigest() for name in assets
    }
    manifest = {
        "schema_version": "roadsense.pages/v1",
        "application_version": __version__,
        "source_commit": source_commit,
        "source_tree_state": source_tree_state,
        "runtime": "deterministic_geometric_fixture",
        "evidence_level": "fixture",
        "evaluation_authorized": False,
        "frozen": False,
        "benchmark_claim_available": False,
        "fixture": {
            "fixture_id": payload["fixture_id"],
            "frame_count": len(payload["frames"]),
            "canvas": payload["canvas"],
            "cadence_ms": payload["cadence_ms"],
            "payload_sha256": payload_hash,
        },
        "assets": assets,
        "asset_sha256": asset_hashes,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("dist/pages"))
    args = parser.parse_args()
    print(build(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
