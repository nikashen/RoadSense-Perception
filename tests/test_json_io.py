from __future__ import annotations

import json

import pytest

from roadsense.json_io import (
    canonical_json_bytes,
    canonical_sha256,
    load_strict_json,
    loads_strict_json,
    write_json_atomic,
)


def test_strict_json_rejects_duplicate_keys() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        loads_strict_json('{"a": 1, "a": 2}')


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_strict_json_rejects_non_finite_numbers(constant: str) -> None:
    with pytest.raises(ValueError, match="non-finite"):
        loads_strict_json(f'{{"value": {constant}}}')


def test_strict_json_rejects_invalid_utf8() -> None:
    with pytest.raises(UnicodeDecodeError):
        loads_strict_json(b"\xff")


def test_canonical_json_is_key_order_independent() -> None:
    assert canonical_json_bytes({"b": 2, "a": 1}) == canonical_json_bytes({"a": 1, "b": 2})
    assert canonical_sha256({"b": 2, "a": 1}) == canonical_sha256({"a": 1, "b": 2})


def test_atomic_json_round_trip(tmp_path) -> None:
    path = write_json_atomic(tmp_path / "report.json", {"status": "ok", "value": 1.5})
    assert path == (tmp_path / "report.json").resolve()
    assert load_strict_json(path) == {"status": "ok", "value": 1.5}
    assert json.loads(path.read_text(encoding="utf-8"))["status"] == "ok"
