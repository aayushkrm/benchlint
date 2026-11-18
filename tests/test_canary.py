"""Tests for benchlint.checks.canary: insert a GUID canary, scan corpora for it."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest

from benchlint.checks.canary import (
    add_canary,
    canary_text,
    new_guid,
    scan_canary,
)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    """Write ``rows`` as JSONL to ``path``."""
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _rows(n: int) -> list[dict[str, object]]:
    """Return ``n`` small dataset rows."""
    return [{"id": i, "question": f"What is {i} plus {i}?", "answer": str(2 * i)} for i in range(n)]


def test_new_guid_seeded_is_reproducible_uuid4() -> None:
    """A seeded GUID is reproducible, a valid UUID and version 4."""
    first = new_guid(seed=7)
    assert first == new_guid(seed=7)
    assert first != new_guid(seed=8)
    parsed = uuid.UUID(first)
    assert parsed.version == 4
    assert str(parsed) == first


def test_new_guid_unseeded_is_fresh() -> None:
    """Without a seed each call gives a new UUID4."""
    assert new_guid() != new_guid()


def test_add_then_scan_round_trip(tmp_path: Path) -> None:
    """A canary added to every row is found on every line of the output, not the input."""
    src = tmp_path / "eval.jsonl"
    dst = tmp_path / "eval_canary.jsonl"
    _write_jsonl(src, _rows(5))

    guid = add_canary(src, dst, guid=new_guid(seed=1))

    assert guid == new_guid(seed=1)
    out_rows = [json.loads(line) for line in dst.read_text(encoding="utf-8").splitlines()]
    assert len(out_rows) == 5
    for original, row in zip(_rows(5), out_rows, strict=True):
        assert row["canary"] == canary_text(guid)
        assert {k: row[k] for k in original} == original  # other fields untouched

    assert scan_canary([dst], guid) == [(dst, 1), (dst, 2), (dst, 3), (dst, 4), (dst, 5)]
    assert scan_canary([src], guid) == []


def test_add_canary_generates_guid_when_none(tmp_path: Path) -> None:
    """With no GUID given, a fresh UUID4 is generated and returned."""
    src = tmp_path / "in.jsonl"
    dst = tmp_path / "out.jsonl"
    _write_jsonl(src, _rows(2))
    guid = add_canary(src, dst)
    assert uuid.UUID(guid).version == 4
    assert len(scan_canary([dst], guid)) == 2


def test_guid_is_canonicalised_and_scan_is_case_insensitive(tmp_path: Path) -> None:
    """An upper-case GUID is returned and matched in canonical lowercase form."""
    src = tmp_path / "in.jsonl"
    dst = tmp_path / "out.jsonl"
    _write_jsonl(src, _rows(1))
    guid = add_canary(src, dst, guid=new_guid(seed=2).upper())
    assert guid == new_guid(seed=2)
    assert scan_canary([dst], new_guid(seed=2).upper()) == [(dst, 1)]


def test_scan_finds_canary_in_plain_text_file(tmp_path: Path) -> None:
    """A text corpus with the canary on one line is reported with that line number."""
    guid = new_guid(seed=3)
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(f"first line\nsecond\n{canary_text(guid)}\nlast\n", encoding="utf-8")
    clean = tmp_path / "clean.txt"
    clean.write_text("nothing here\n", encoding="utf-8")
    assert scan_canary([corpus, clean], guid) == [(corpus, 3)]


def test_scan_missing_guid_gives_no_hits(tmp_path: Path) -> None:
    """A corpus that does not contain the canary gives no hits."""
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("ordinary text\n", encoding="utf-8")
    assert scan_canary([corpus], new_guid(seed=4)) == []


def test_empty_file_round_trip(tmp_path: Path) -> None:
    """An empty dataset gets an output file with no rows, and scanning it finds nothing."""
    src = tmp_path / "empty.jsonl"
    dst = tmp_path / "empty_out.jsonl"
    src.write_text("", encoding="utf-8")
    guid = add_canary(src, dst, guid=new_guid(seed=5))
    assert dst.read_text(encoding="utf-8") == ""
    assert scan_canary([dst], guid) == []


def test_invalid_guid_is_rejected(tmp_path: Path) -> None:
    """A string that is not a GUID is an error for both add and scan."""
    src = tmp_path / "in.jsonl"
    _write_jsonl(src, _rows(1))
    with pytest.raises(ValueError, match="not a valid GUID"):
        add_canary(src, tmp_path / "out.jsonl", guid="not-a-guid")
    with pytest.raises(ValueError, match="not a valid GUID"):
        scan_canary([src], "12345")


def test_dst_equal_to_src_is_rejected(tmp_path: Path) -> None:
    """Writing over the input would destroy it, so the call is refused."""
    src = tmp_path / "in.jsonl"
    _write_jsonl(src, _rows(1))
    with pytest.raises(ValueError, match="must differ"):
        add_canary(src, src, guid=new_guid(seed=6))
    assert json.loads(src.read_text(encoding="utf-8").splitlines()[0])["id"] == 0


def test_existing_field_is_not_overwritten(tmp_path: Path) -> None:
    """A row that already has the canary field is an error, not a silent overwrite."""
    src = tmp_path / "in.jsonl"
    dst = tmp_path / "out.jsonl"
    _write_jsonl(src, [{"id": 0, "canary": "keep me"}])
    with pytest.raises(ValueError, match="already present"):
        add_canary(src, dst, guid=new_guid(seed=9))


def test_non_object_row_is_rejected(tmp_path: Path) -> None:
    """Each JSONL line must be an object."""
    src = tmp_path / "in.jsonl"
    src.write_text("[1, 2, 3]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected a JSON object"):
        add_canary(src, tmp_path / "out.jsonl", guid=new_guid(seed=10))
