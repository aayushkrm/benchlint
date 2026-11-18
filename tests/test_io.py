"""Loader tests on local fixtures written to tmp_path. No network.

The Hugging Face tests stub ``datasets.load_dataset`` and ``huggingface_hub`` with local data.
One real-network test is skipped unless ``BENCHLINT_NETWORK_TESTS=1`` is set.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

import huggingface_hub
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from benchlint.io import load_dataset, load_dataset_detailed
from benchlint.schema import DatasetSpec, SourceSpec


def _spec(
    source: SourceSpec,
    *,
    text_fields: list[str] | None = None,
    label_field: str | None = "label",
    id_field: str | None = "id",
    max_rows: int | None = None,
    seed: int = 0,
) -> DatasetSpec:
    return DatasetSpec(
        name="fixture",
        source=source,
        text_fields=text_fields if text_fields is not None else ["text"],
        label_field=label_field,
        id_field=id_field,
        max_rows_per_split=max_rows,
        seed=seed,
    )


def _local(kind: str, path: str, split_map: dict[str, str]) -> SourceSpec:
    return SourceSpec(
        kind=kind,  # type: ignore[arg-type]
        id_or_path=path,
        config=None,
        revision=None,
        split_map=split_map,
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _rows(n: int, prefix: str = "item") -> list[dict[str, Any]]:
    return [{"id": f"{prefix}{i}", "text": f"text number {i}", "label": i % 3} for i in range(n)]


# --- local loaders -------------------------------------------------------------------------


def test_jsonl_split_map_direction_and_fields(tmp_path: Path) -> None:
    _write_jsonl(tmp_path / "validation_matched.jsonl", _rows(3, "v"))
    _write_jsonl(tmp_path / "train.jsonl", _rows(2, "t"))
    spec = _spec(
        _local(
            "jsonl",
            str(tmp_path),
            {"validation": "validation_matched", "train": "train"},
        ),
        text_fields=["text"],
    )
    out = load_dataset(spec)
    # Keys are benchlint names; the file read is the source name.
    assert list(out) == ["validation", "train"]
    assert [r.id for r in out["validation"]] == ["v0", "v1", "v2"]
    assert all(r.split == "validation" for r in out["validation"])
    first = out["validation"][0]
    assert first.fields == {"text": "text number 0"}  # id and label are not in fields
    assert first.label == 0
    assert out["train"][1].label == 1


def test_jsonl_relative_path_uses_base_dir(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    _write_jsonl(data / "test.jsonl", _rows(2))
    spec = _spec(_local("jsonl", "data/{split}.jsonl", {"test": "test"}))
    out = load_dataset(spec, base_dir=tmp_path)
    assert len(out["test"]) == 2


def test_jsonl_empty_file_gives_empty_split(tmp_path: Path) -> None:
    (tmp_path / "test.jsonl").write_text("", encoding="utf-8")
    out = load_dataset(_spec(_local("jsonl", str(tmp_path / "test.jsonl"), {"test": "test"})))
    assert out == {"test": []}


def test_jsonl_unicode_and_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "test.jsonl"
    path.write_text(
        '{"id": "a", "text": "日本語 café", "label": "ja"}\n\n'
        '{"id": "b", "text": "Привет", "label": "ru"}\n',
        encoding="utf-8",
    )
    out = load_dataset(_spec(_local("jsonl", str(path), {"test": "test"})))
    assert [r.fields["text"] for r in out["test"]] == ["日本語 café", "Привет"]
    assert [r.label for r in out["test"]] == ["ja", "ru"]


def test_jsonl_non_object_line_is_an_error(tmp_path: Path) -> None:
    path = tmp_path / "test.jsonl"
    path.write_text('["not", "an", "object"]\n', encoding="utf-8")
    with pytest.raises(ValueError, match="expected a JSON object"):
        load_dataset(_spec(_local("jsonl", str(path), {"test": "test"})))


def test_csv_loader_with_bom_and_empty_label(tmp_path: Path) -> None:
    path = tmp_path / "test.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["id", "text", "label"])
        writer.writerow(["c1", "Hello, wörld", "pos"])
        writer.writerow(["c2", "", ""])
    out = load_dataset(_spec(_local("csv", str(path), {"test": "test"})))
    assert [r.id for r in out["test"]] == ["c1", "c2"]
    assert out["test"][0].fields == {"text": "Hello, wörld"}
    assert out["test"][0].label == "pos"
    assert out["test"][1].label is None  # empty label cell becomes None


def test_csv_header_only_is_empty(tmp_path: Path) -> None:
    path = tmp_path / "test.csv"
    path.write_text("id,text,label\n", encoding="utf-8")
    assert load_dataset(_spec(_local("csv", str(path), {"test": "test"}))) == {"test": []}


def test_parquet_loader_and_cap(tmp_path: Path) -> None:
    rows = _rows(40, "p")
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, tmp_path / "train.parquet")
    spec = _spec(_local("parquet", str(tmp_path), {"train": "train"}), max_rows=7, seed=3)
    records, loads = load_dataset_detailed(spec)
    assert len(records["train"]) == 7
    assert loads[0].backend == "parquet"
    assert loads[0].source_rows == 40
    assert loads[0].kept_rows == 7
    assert loads[0].sampled is True
    assert all(r.fields["text"].startswith("text number") for r in records["train"])


def test_parquet_fields_include_non_text_columns_as_text(tmp_path: Path) -> None:
    table = pa.Table.from_pylist(
        [{"id": "x", "text": "hi", "label": 1, "score": 2.5, "tags": ["a", "b"], "n": None}]
    )
    pq.write_table(table, tmp_path / "test.parquet")
    out = load_dataset(_spec(_local("parquet", str(tmp_path), {"test": "test"})))
    rec = out["test"][0]
    assert rec.fields == {"text": "hi", "score": "2.5", "tags": '["a", "b"]', "n": ""}
    assert rec.label == 1


# --- ids, cap and determinism ---------------------------------------------------------------


def test_default_id_uses_source_row_index_and_survives_cap(tmp_path: Path) -> None:
    rows = [{"text": f"row {i}", "label": 0} for i in range(30)]
    _write_jsonl(tmp_path / "test.jsonl", rows)
    spec = _spec(
        _local("jsonl", str(tmp_path / "test.jsonl"), {"test": "test"}),
        id_field=None,
        max_rows=10,
        seed=5,
    )
    out = load_dataset(spec)
    ids = [r.id for r in out["test"]]
    indices = [int(i.split(":")[1]) for i in ids]
    assert indices == sorted(indices)  # sampled rows keep source order
    assert len(set(ids)) == 10
    for rec, idx in zip(out["test"], indices, strict=True):
        assert rec.fields["text"] == f"row {idx}"  # the id points at the right source row


def test_cap_is_deterministic_for_same_seed_and_varies_with_seed(tmp_path: Path) -> None:
    _write_jsonl(tmp_path / "test.jsonl", _rows(200))
    source = _local("jsonl", str(tmp_path / "test.jsonl"), {"test": "test"})
    first = load_dataset(_spec(source, max_rows=20, seed=11))["test"]
    again = load_dataset(_spec(source, max_rows=20, seed=11))["test"]
    other = load_dataset(_spec(source, max_rows=20, seed=12))["test"]
    assert [r.id for r in first] == [r.id for r in again]
    assert [r.id for r in first] != [r.id for r in other]
    assert len(first) == 20


def test_cap_larger_than_split_keeps_everything(tmp_path: Path) -> None:
    _write_jsonl(tmp_path / "test.jsonl", _rows(5))
    records, loads = load_dataset_detailed(
        _spec(_local("jsonl", str(tmp_path / "test.jsonl"), {"test": "test"}), max_rows=50)
    )
    assert len(records["test"]) == 5
    assert loads[0].sampled is False


def test_cap_below_one_is_rejected(tmp_path: Path) -> None:
    _write_jsonl(tmp_path / "test.jsonl", _rows(3))
    with pytest.raises(ValueError, match="at least 1"):
        load_dataset(
            _spec(_local("jsonl", str(tmp_path / "test.jsonl"), {"test": "test"}), max_rows=0)
        )


def test_split_seed_is_per_split(tmp_path: Path) -> None:
    _write_jsonl(tmp_path / "a.jsonl", _rows(100, "a"))
    _write_jsonl(tmp_path / "b.jsonl", _rows(100, "b"))
    spec = _spec(
        _local("jsonl", str(tmp_path / "{split}.jsonl"), {"a": "a", "b": "b"}),
        max_rows=10,
        seed=0,
    )
    out = load_dataset(spec)
    a_pos = [int(r.id[1:]) for r in out["a"]]
    b_pos = [int(r.id[1:]) for r in out["b"]]
    assert a_pos != b_pos  # independent draws per split, same seed


def test_empty_split_map_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="split_map is empty"):
        load_dataset(_spec(_local("jsonl", str(tmp_path), {})))


def test_missing_text_field_names_the_split(tmp_path: Path) -> None:
    _write_jsonl(tmp_path / "test.jsonl", [{"id": "a", "body": "x", "label": 0}])
    with pytest.raises(ValueError, match="split 'test': text field 'text'"):
        load_dataset(_spec(_local("jsonl", str(tmp_path / "test.jsonl"), {"test": "test"})))


def test_single_file_with_two_splits_is_rejected(tmp_path: Path) -> None:
    _write_jsonl(tmp_path / "all.jsonl", _rows(2))
    spec = _spec(_local("jsonl", str(tmp_path / "all.jsonl"), {"train": "a", "test": "b"}))
    with pytest.raises(ValueError, match="single file"):
        load_dataset(spec)


def test_missing_file_is_a_file_not_found_error(tmp_path: Path) -> None:
    spec = _spec(_local("jsonl", str(tmp_path / "nope.jsonl"), {"test": "test"}))
    with pytest.raises(FileNotFoundError):
        load_dataset(spec)


# --- Hugging Face (offline stubs) -----------------------------------------------------------


def _hf_source(split_map: dict[str, str]) -> SourceSpec:
    return SourceSpec(
        kind="hf",
        id_or_path="org/fixture",
        config="main",
        revision=None,
        split_map=split_map,
    )


def test_hf_path_uses_datasets_with_source_split_names(monkeypatch: pytest.MonkeyPatch) -> None:
    import datasets

    calls: list[dict[str, Any]] = []
    rows_by_split = {
        "validation_matched": [{"id": "m0", "text": "a", "label": 1}],
        "train": [{"id": "t0", "text": "b", "label": 0}, {"id": "t1", "text": "c", "label": 1}],
    }

    def fake_load(path: str, name: str | None = None, split: str | None = None, **kw: Any) -> Any:
        calls.append({"path": path, "name": name, "split": split, **kw})
        assert split is not None
        return datasets.Dataset.from_list(rows_by_split[split])

    monkeypatch.setattr(datasets, "load_dataset", fake_load)
    spec = _spec(_hf_source({"validation": "validation_matched", "train": "train"}))
    records, loads = load_dataset_detailed(spec)

    assert [c["split"] for c in calls] == ["validation_matched", "train"]
    assert calls[0]["name"] == "main"
    assert [r.id for r in records["validation"]] == ["m0"]
    assert [r.id for r in records["train"]] == ["t0", "t1"]
    assert {ld.backend for ld in loads} == {"hf-datasets"}


def test_hf_fallback_reads_parquet_convert_branch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import datasets

    def failing_load(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("Dataset scripts are no longer supported")

    monkeypatch.setattr(datasets, "load_dataset", failing_load)

    shard = tmp_path / "shard.parquet"
    pq.write_table(
        pa.Table.from_pylist([{"id": "p0", "text": "fallback", "label": 2}]),
        shard,
    )
    seen_revisions: set[str] = set()

    def fake_list(repo_id: str, *, repo_type: str, revision: str) -> list[str]:
        seen_revisions.add(revision)
        assert repo_id == "org/fixture"
        return ["main/test/0000.parquet", "main/train/0000.parquet", "other/test/0000.parquet"]

    def fake_download(repo_id: str, filename: str, *, repo_type: str, revision: str) -> str:
        seen_revisions.add(revision)
        assert filename == "main/test/0000.parquet"
        return str(shard)

    monkeypatch.setattr(huggingface_hub, "list_repo_files", fake_list)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)

    records, loads = load_dataset_detailed(_spec(_hf_source({"test": "test"})))
    assert seen_revisions == {"refs/convert/parquet"}
    assert [r.id for r in records["test"]] == ["p0"]
    assert records["test"][0].label == 2
    assert loads[0].backend == "hf-parquet-fallback"


def test_hf_fallback_without_matching_files_fails_clearly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import datasets

    def failing_load(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("boom")

    monkeypatch.setattr(datasets, "load_dataset", failing_load)
    monkeypatch.setattr(
        huggingface_hub, "list_repo_files", lambda *a, **k: ["main/train/0000.parquet"]
    )
    with pytest.raises(FileNotFoundError, match="no parquet files for main/test"):
        load_dataset(_spec(_hf_source({"test": "test"})))


@pytest.mark.skipif(
    os.environ.get("BENCHLINT_NETWORK_TESTS") != "1",
    reason="network test: set BENCHLINT_NETWORK_TESTS=1 to run",
)
def test_network_gsm8k_small_sample() -> None:
    spec = DatasetSpec(
        name="gsm8k",
        source=SourceSpec(
            kind="hf",
            id_or_path="openai/gsm8k",
            config="main",
            revision=None,
            split_map={"test": "test"},
        ),
        text_fields=["question"],
        label_field=None,
        id_field=None,
        max_rows_per_split=5,
        seed=0,
    )
    out = load_dataset(spec)
    assert len(out["test"]) == 5


def test_drop_label_values_removes_rows_before_the_cap(tmp_path: Path) -> None:
    """Rows with a dropped label never reach the sample, so the cap counts kept rows only."""
    from benchlint.io import load_dataset_detailed
    from benchlint.schema import DatasetSpec, SourceSpec

    rows = [{"id": str(i), "text": f"t{i}", "label": -1 if i % 2 else 0} for i in range(40)]
    path = tmp_path / "data.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    spec = DatasetSpec(
        name="drop",
        source=SourceSpec(
            kind="jsonl",
            id_or_path=str(path),
            config=None,
            revision=None,
            split_map={"train": "data"},
        ),
        text_fields=["text"],
        label_field="label",
        id_field="id",
        max_rows_per_split=10,
        seed=0,
        drop_label_values=[-1],
    )
    records, loads = load_dataset_detailed(spec)
    assert len(records["train"]) == 10
    assert all(r.label == 0 for r in records["train"])
    assert loads[0].dropped_by_label == 20
    assert loads[0].source_rows == 40
    assert loads[0].kept_rows == 10


def test_without_drop_values_nothing_is_removed(tmp_path: Path) -> None:
    from benchlint.io import load_dataset_detailed
    from benchlint.schema import DatasetSpec, SourceSpec

    path = tmp_path / "data.jsonl"
    path.write_text('{"id": "a", "text": "x", "label": -1}\n', encoding="utf-8")
    spec = DatasetSpec(
        name="keep",
        source=SourceSpec(
            kind="jsonl",
            id_or_path=str(path),
            config=None,
            revision=None,
            split_map={"train": "data"},
        ),
        text_fields=["text"],
        label_field="label",
        id_field="id",
        max_rows_per_split=None,
        seed=0,
    )
    records, loads = load_dataset_detailed(spec)
    assert len(records["train"]) == 1
    assert loads[0].dropped_by_label == 0
