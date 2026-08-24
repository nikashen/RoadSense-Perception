"""Verify a built Pages artifact without importing optional runtime dependencies."""

from __future__ import annotations

import argparse
import hashlib
import json
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse


class _AssetReferenceParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.references: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "script" and attributes.get("src"):
            self.references.append(attributes["src"] or "")
        if tag == "link" and attributes.get("href"):
            self.references.append(attributes["href"] or "")


def _reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is not allowed: {value}")


def _loads_strict(raw: str) -> object:
    return json.loads(
        raw,
        object_pairs_hook=_reject_duplicate_pairs,
        parse_constant=_reject_constant,
    )


def verify(root: Path) -> dict[str, object]:
    root = root.resolve()
    if not root.is_dir():
        raise ValueError(f"Pages directory does not exist: {root}")
    manifest_path = root / "manifest.json"
    try:
        manifest = _loads_strict(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Pages manifest is unreadable: {exc}") from exc
    if not isinstance(manifest, dict):
        raise TypeError("Pages manifest must be a JSON object")
    expected_flags = {
        "schema_version": "roadsense.pages/v1",
        "runtime": "deterministic_geometric_fixture",
        "evidence_level": "fixture",
        "evaluation_authorized": False,
        "frozen": False,
        "benchmark_claim_available": False,
    }
    for key, expected in expected_flags.items():
        if manifest.get(key) != expected:
            raise ValueError(f"Pages manifest has invalid {key}")

    assets = manifest.get("assets")
    asset_hashes = manifest.get("asset_sha256")
    if (
        not isinstance(assets, list)
        or not assets
        or any(not isinstance(name, str) for name in assets)
        or len(set(assets)) != len(assets)
    ):
        raise ValueError("Pages manifest assets must be a non-empty unique list")
    expected_assets = {
        "index.html",
        "app.js",
        "styles.css",
        "favicon.svg",
        "demo.json",
        ".nojekyll",
    }
    if set(assets) != expected_assets:
        raise ValueError("Pages manifest assets do not match the required artifact set")
    if not isinstance(asset_hashes, dict) or set(asset_hashes) != set(assets):
        raise ValueError("Pages manifest asset hashes do not match assets")
    for name in assets:
        if (
            not isinstance(name, str)
            or not name
            or Path(name).name != name
            or "/" in name
            or "\\" in name
        ):
            raise ValueError(f"unsafe Pages asset name: {name!r}")
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Pages asset is missing: {name}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if asset_hashes[name] != digest:
            raise ValueError(f"Pages asset hash mismatch: {name}")
    root_entries = {entry.name for entry in root.iterdir()}
    if root_entries != expected_assets | {"manifest.json"}:
        raise ValueError("Pages directory contains unexpected files")

    parser = _AssetReferenceParser()
    try:
        parser.feed((root / "index.html").read_text(encoding="utf-8"))
        parser.close()
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"Pages index is unreadable: {exc}") from exc
    references = set(parser.references)
    required_references = {"app.js", "styles.css", "favicon.svg"}
    if references != required_references:
        raise ValueError(
            f"Pages index references {sorted(references)!r}; expected "
            f"{sorted(required_references)!r}"
        )
    for reference in references:
        parsed = urlparse(reference)
        if parsed.scheme or parsed.netloc or reference.startswith("/"):
            raise ValueError(f"Pages asset reference must be relative: {reference}")

    fixture = manifest.get("fixture")
    if not isinstance(fixture, dict):
        raise TypeError("Pages manifest fixture metadata is missing")
    if fixture.get("frame_count") != 24 or fixture.get("cadence_ms") != 100:
        raise ValueError("Pages fixture metadata is inconsistent")
    if fixture.get("canvas") != {"width": 960, "height": 540}:
        raise ValueError("Pages canvas metadata is inconsistent")
    try:
        payload = _loads_strict((root / "demo.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Pages demo payload is unreadable: {exc}") from exc
    if not isinstance(payload, dict):
        raise TypeError("Pages demo payload must be a JSON object")
    payload_hash = hashlib.sha256(
        (
            json.dumps(
                payload,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
    ).hexdigest()
    if fixture.get("payload_sha256") != payload_hash:
        raise ValueError("Pages demo payload hash does not match manifest")
    if payload.get("schema_version") != "roadsense.demo/v1":
        raise ValueError("Pages demo payload schema is invalid")
    if payload.get("source") != "deterministic_geometric_fixture":
        raise ValueError("Pages demo payload source is invalid")
    if payload.get("fixture_id") != fixture.get("fixture_id"):
        raise ValueError("Pages fixture ID does not match demo payload")
    frames = payload.get("frames")
    if not isinstance(frames, list):
        raise TypeError("Pages demo payload frames must be a list")
    if len(frames) != fixture.get("frame_count"):
        raise ValueError("Pages frame count does not match demo payload")
    if payload.get("canvas") != fixture.get("canvas") or payload.get("cadence_ms") != fixture.get(
        "cadence_ms"
    ):
        raise ValueError("Pages fixture metadata does not match demo payload")
    evidence = payload.get("evidence")
    if (
        not isinstance(evidence, dict)
        or evidence.get("level") != "fixture"
        or evidence.get("evaluation_authorized") is not False
        or evidence.get("frozen") is not False
        or evidence.get("benchmark_claim_available") is not False
    ):
        raise ValueError("Pages demo payload evidence is invalid")
    return {
        "status": "ok",
        "directory": str(root),
        "assets": len(assets),
        "fixture_id": fixture.get("fixture_id"),
        "frame_count": fixture.get("frame_count"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify a RoadSense Pages artifact")
    parser.add_argument("directory", type=Path, nargs="?", default=Path("dist/pages"))
    args = parser.parse_args()
    print(json.dumps(verify(args.directory), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
