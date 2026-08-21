"""Build a relative-path GitHub Pages artifact for the deterministic fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def build(output: Path) -> Path:
    root = Path(__file__).resolve().parents[1]
    source = root / "src" / "roadsense" / "web"
    required = ("index.html", "app.js", "styles.css", "favicon.svg")
    if any(not (source / name).is_file() for name in required):
        raise FileNotFoundError("web assets are incomplete")
    output = output.resolve()
    pages_root = (root / "dist").resolve()
    if output == pages_root or pages_root not in output.parents:
        raise ValueError("Pages output must be a child of repository dist/")
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    for name in required:
        shutil.copy2(source / name, output / name)
    (output / ".nojekyll").write_text("", encoding="ascii")
    assets = list(required) + [".nojekyll"]
    asset_hashes = {
        name: hashlib.sha256((output / name).read_bytes()).hexdigest() for name in assets
    }
    fixture_manifest = json.loads(
        (root / "configs" / "fixture_manifest.json").read_text(encoding="utf-8")
    )
    manifest = {
        "schema_version": "roadsense.pages/v1",
        "runtime": "deterministic_geometric_fixture",
        "evidence_level": "fixture",
        "evaluation_authorized": False,
        "frozen": False,
        "benchmark_claim_available": False,
        "fixture": {
            "fixture_id": "roadsense-city-loop-v1",
            "frame_count": 24,
            "canvas": {"width": 960, "height": 540},
            "cadence_ms": 100,
            "payload_sha256": fixture_manifest["content_sha256"],
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
