"""Command-line entry points for RoadSense-Perception."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, cast

from roadsense.contracts import DatasetManifest
from roadsense.evidence import build_fixture_report
from roadsense.fixture import build_demo_payload
from roadsense.json_io import load_strict_json, write_json_atomic


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="roadsense",
        description="Evidence-first detection, segmentation, and tracking laboratory",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    smoke = subparsers.add_parser("smoke", help="run deterministic perception fixture")
    smoke.add_argument("--json", action="store_true", dest="as_json")

    report = subparsers.add_parser("report", help="write deterministic fixture evidence")
    report.add_argument("--output", type=Path, default=Path("reports/fixture_v1.json"))

    audit = subparsers.add_parser("audit-manifest", help="strictly validate a dataset manifest")
    audit.add_argument("manifest", type=Path)

    serve = subparsers.add_parser("serve", help="run the local Perception Workbench")
    serve.add_argument("--host", default=os.getenv("ROADSENSE_HOST", "127.0.0.1"))
    serve.add_argument("--port", type=int, default=8100)
    return parser


def _run_smoke(as_json: bool) -> int:
    payload = build_demo_payload()
    frames = cast(list[dict[str, Any]], payload["frames"])
    metrics = cast(dict[str, dict[str, Any]], payload["metrics"])
    summary = {
        "status": "ok",
        "fixture_id": payload["fixture_id"],
        "frames": len(frames),
        "detection_ap50": metrics["detection"]["ap"],
        "segmentation_mean_iou": metrics["segmentation"]["mean_iou"],
        "tracking_mota": metrics["tracking"]["mota"],
        "benchmark_claim_available": False,
    }
    print(
        json.dumps(summary, ensure_ascii=False, allow_nan=False, sort_keys=True)
        if as_json
        else "RoadSense fixture OK: "
        + ", ".join(f"{key}={value}" for key, value in summary.items())
    )
    return 0


def _run_serve(host: str, port: int) -> int:
    if not 1 <= port <= 65_535:
        print("port must be in [1, 65535]", file=sys.stderr)
        return 2
    try:
        import uvicorn

        from roadsense.api import create_app
    except ImportError:
        print("serve requires roadsense-perception[serve]", file=sys.stderr)
        return 2
    uvicorn.run(create_app(), host=host, port=port, log_level="info")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "smoke":
        return _run_smoke(args.as_json)
    if args.command == "report":
        path = write_json_atomic(args.output, build_fixture_report())
        print(path)
        return 0
    if args.command == "audit-manifest":
        try:
            manifest = DatasetManifest.model_validate(load_strict_json(args.manifest))
        except (OSError, UnicodeError, ValueError) as exc:
            print(f"manifest audit failed: {exc}", file=sys.stderr)
            return 2
        print(manifest.model_dump_json(indent=2))
        return 0
    if args.command == "serve":
        return _run_serve(args.host, args.port)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
