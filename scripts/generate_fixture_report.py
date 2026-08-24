from __future__ import annotations

import argparse
from pathlib import Path

from roadsense.evidence import build_fixture_report
from roadsense.json_io import write_json_atomic


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("reports/fixture_v1.json"))
    args = parser.parse_args()
    print(write_json_atomic(args.output, build_fixture_report()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
