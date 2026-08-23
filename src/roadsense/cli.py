"""Command-line entry points for RoadSense-Perception."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, cast

from roadsense.adapters import (
    ArtifactVerificationError,
    load_artifact_manifest,
    verify_artifact_manifest,
)
from roadsense.contracts import DatasetManifest
from roadsense.evidence import build_fixture_report
from roadsense.fixture import build_demo_payload
from roadsense.json_io import canonical_sha256, load_strict_json, write_json_atomic
from roadsense.local_eval import (
    LocalEvaluationError,
    evaluate_local,
    fixture_dry_run_summary,
)
from roadsense.runtime import build_fixture_runtime_record


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

    benchmark = subparsers.add_parser(
        "benchmark",
        aliases=["runtime-audit"],
        help="record a fixture runtime audit (not a model FPS benchmark)",
    )
    benchmark.add_argument("--output", type=Path, default=Path("reports/runtime_fixture_v1.json"))
    benchmark.add_argument("--iterations", type=int, default=1)
    benchmark.add_argument("--json", action="store_true", dest="as_json")

    audit = subparsers.add_parser("audit-manifest", help="strictly validate a dataset manifest")
    audit.add_argument("manifest", type=Path)

    artifact_audit = subparsers.add_parser(
        "audit-artifact",
        aliases=["verify-artifact"],
        help="validate a local model-artifact manifest and optionally verify its file hash",
    )
    artifact_audit.add_argument("manifest", type=Path)
    artifact_audit.add_argument(
        "--root",
        type=Path,
        help="allow-listed artifact root; omit for manifest-only dry-run (no model is loaded)",
    )
    artifact_audit.add_argument("--output", type=Path, help="write the JSON verification receipt")
    artifact_audit.add_argument("--json", action="store_true", dest="as_json")

    local_eval = subparsers.add_parser(
        "evaluate-local",
        help="evaluate explicitly provided local sequence data (never downloads data)",
    )
    local_eval.add_argument(
        "spec",
        type=Path,
        nargs="?",
        help="roadsense.local-evaluation/v1 JSON spec",
    )
    local_eval.add_argument(
        "--fixture",
        action="store_true",
        help="run the deterministic fixture dry-run instead of reading local data",
    )
    local_eval.add_argument("--output", type=Path, help="write the hash-bound report JSON")
    local_eval.add_argument("--json", action="store_true", dest="as_json")

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


def _run_evaluate_local(
    spec: Path | None,
    *,
    fixture: bool,
    output: Path | None,
    as_json: bool,
) -> int:
    if fixture and spec is not None:
        print("evaluate-local accepts either a spec path or --fixture, not both", file=sys.stderr)
        return 2
    if not fixture and spec is None:
        print(
            "a local evaluation spec is required; use --fixture for the explicit synthetic dry-run",
            file=sys.stderr,
        )
        return 2
    try:
        if fixture:
            summary = fixture_dry_run_summary()
            if output is not None:
                # Keep the persisted artifact compatible with the report API;
                # the stdout summary still exposes the explicit fixture flags.
                path = write_json_atomic(output, build_fixture_report())
                if not as_json:
                    print(path)
                    return 0
            print(
                json.dumps(summary, ensure_ascii=False, allow_nan=False, sort_keys=True)
                if as_json
                else "RoadSense local fixture dry-run: "
                + ", ".join(f"{key}={value}" for key, value in summary.items())
            )
            return 0
        report = evaluate_local(cast(Path, spec))
        if output is not None:
            path = write_json_atomic(output, report)
            if not as_json:
                print(path)
                return 0
        print(json.dumps(report, ensure_ascii=False, allow_nan=False, sort_keys=True))
        return 0
    except (LocalEvaluationError, OSError, UnicodeError, TypeError, ValueError, MemoryError) as exc:
        print(f"local evaluation failed: {exc}", file=sys.stderr)
        return 2


def _run_benchmark(output: Path, iterations: int, as_json: bool) -> int:
    try:
        record = build_fixture_runtime_record(iterations=iterations)
        path = write_json_atomic(output, record.model_dump(mode="json"))
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        print(f"runtime audit failed: {exc}", file=sys.stderr)
        return 2
    if as_json:
        print(record.model_dump_json(indent=2))
    else:
        print(path)
    return 0


def _run_audit_artifact(
    manifest_path: Path,
    *,
    root: Path | None,
    output: Path | None,
    as_json: bool,
) -> int:
    """Audit an artifact manifest without importing or loading a model.

    With no ``--root`` this is an intentional schema/hash dry-run. Supplying
    ``--root`` additionally hashes the checkpoint and optional dependency
    lock, returning an immutable verification receipt.
    """

    try:
        manifest = load_artifact_manifest(manifest_path)
        manifest_payload = manifest.model_dump(mode="json")
        if root is None:
            result: dict[str, object] = {
                "status": "manifest_valid",
                "verification_scope": "manifest_only",
                "manifest_sha256": canonical_sha256(manifest_payload),
                "artifact_id": manifest.artifact_id,
                "artifact_path": manifest.artifact_path,
                "artifact_sha256": manifest.artifact_sha256,
                "model_loaded": False,
                "verified": False,
            }
        else:
            receipt = verify_artifact_manifest(manifest, artifact_root=root)
            result = {
                "status": "verified",
                "verification_scope": "manifest_and_artifact",
                **receipt.model_dump(mode="json"),
                "model_loaded": False,
            }
        if output is not None:
            path = write_json_atomic(output, result)
            if not as_json:
                print(path)
                return 0
        print(json.dumps(result, ensure_ascii=False, allow_nan=False, sort_keys=True))
        return 0
    except (ArtifactVerificationError, OSError, UnicodeError, TypeError, ValueError) as exc:
        print(f"artifact audit failed: {exc}", file=sys.stderr)
        return 2


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "smoke":
        return _run_smoke(args.as_json)
    if args.command == "report":
        path = write_json_atomic(args.output, build_fixture_report())
        print(path)
        return 0
    if args.command in {"benchmark", "runtime-audit"}:
        return _run_benchmark(args.output, args.iterations, args.as_json)
    if args.command == "audit-manifest":
        try:
            manifest = DatasetManifest.model_validate(load_strict_json(args.manifest))
        except (OSError, UnicodeError, ValueError) as exc:
            print(f"manifest audit failed: {exc}", file=sys.stderr)
            return 2
        print(manifest.model_dump_json(indent=2))
        return 0
    if args.command in {"audit-artifact", "verify-artifact"}:
        return _run_audit_artifact(
            args.manifest,
            root=args.root,
            output=args.output,
            as_json=args.as_json,
        )
    if args.command == "evaluate-local":
        return _run_evaluate_local(
            args.spec,
            fixture=args.fixture,
            output=args.output,
            as_json=args.as_json,
        )
    if args.command == "serve":
        return _run_serve(args.host, args.port)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
