"""Verify a sanitized BDD100K Detection benchmark receipt offline.

The verifier intentionally accepts only the public receipt.  It never opens
BDD images, labels, predictions, model weights, or evaluator workspaces.  A
successful exit means that the receipt schema, canonical report id, pinned
devkit identity, finite metrics, and the two-run consistency contract are all
valid; it does not download data or reproduce inference.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from roadsense.bdd_benchmark import (
    BDD100KBenchmarkReceiptError,
    load_bdd100k_detection_receipt,
)


def verify_bdd100k_detection_benchmark(path: Path | str) -> dict[str, Any]:
    """Load and validate one sanitized benchmark receipt.

    The returned dictionary is a JSON-compatible normalized representation,
    suitable for a CI summary or a release-note generator.
    """

    receipt = load_bdd100k_detection_receipt(Path(path))
    return receipt.model_dump(mode="json")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path, help="sanitized benchmark-receipt.json")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        payload = verify_bdd100k_detection_benchmark(args.receipt)
    except (BDD100KBenchmarkReceiptError, OSError, TypeError, ValueError, UnicodeError) as exc:
        print(f"BDD100K benchmark receipt verification failed: {exc}", file=sys.stderr)
        return 2
    summary = {
        "status": "ok",
        "report_id": payload["report_id"],
        "scope": payload["scope"],
        "metrics": payload["evaluator_runs"][0]["metrics"],
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "verify_bdd100k_detection_benchmark"]
