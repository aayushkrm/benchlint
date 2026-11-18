"""Dataset loaders: Hugging Face (with a parquet fallback), JSONL, CSV and parquet.

Yields ``Record(id, split, fields, label)``. ``max_rows_per_split`` caps each split with a
seeded sample, so runs are reproducible.

Conventions:

- ``split_map`` maps the benchlint split name (key) to the source split name (value).
  Records carry the benchlint name in ``Record.split``.
- ``fields`` holds every column except the id and label columns, values converted to text.
  Columns named in ``text_fields`` are required; a missing one raises ``ValueError``.
- Without ``id_field`` the id is ``"<split>:<source row index>"``. The index is taken before
  sampling, so an id does not change when the cap changes.
- For local sources ``id_or_path`` is one of: a path containing ``{split}`` (replaced by the
  source split name), a directory holding ``<source split>.<ext>`` files, or a single file
  (only when ``split_map`` names exactly one split). Relative paths resolve against
  ``base_dir``.
- For Hugging Face sources, ``datasets.load_dataset`` is tried first. If it fails, the
  parquet files of ``config/split`` on the ``refs/convert/parquet`` revision are read through
  ``huggingface_hub``. The fallback ignores ``source.revision``. ``SplitLoad.backend`` says
  which path was used.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from benchlint.schema import DatasetSpec, Record, SourceSpec

logger = logging.getLogger(__name__)

PARQUET_REVISION = "refs/convert/parquet"
DEFAULT_CONFIG = "default"
_LOCAL_EXTENSIONS = {"jsonl": ".jsonl", "csv": ".csv", "parquet": ".parquet"}


@dataclass(frozen=True)
class SplitLoad:
    """Provenance of one loaded split, for the audit report.

    ``source_rows`` counts the rows in the source split before the cap. ``kept_rows`` counts
    the rows returned as records.
    """

    split: str
    source_split: str
    backend: str
    source_rows: int
    kept_rows: int
    sampled: bool
    dropped_by_label: int = 0


@dataclass(frozen=True)
class _Source:
    """An opened source split: its backend name, row count, and a row-taking function.

    ``take(None)`` returns every row; ``take(indices)`` returns the rows at those positions,
    in the given order.
    """

    backend: str
    n_rows: int
    take: Callable[[list[int] | None], list[dict[str, Any]]]
    column: Callable[[str], list[Any]]


def load_dataset(spec: DatasetSpec, *, base_dir: Path | None = None) -> dict[str, list[Record]]:
    """Load every split named in ``spec.source.split_map`` and return them keyed by split name.

    ``base_dir`` is used to resolve relative paths for local (jsonl, csv, parquet) sources;
    the audit CLI passes the directory that holds the spec file.
    Hugging Face sources try ``datasets`` first and fall back to parquet files from the
    ``refs/convert/parquet`` revision via ``huggingface_hub``.
    """
    records, _ = load_dataset_detailed(spec, base_dir=base_dir)
    return records


def load_dataset_detailed(
    spec: DatasetSpec, *, base_dir: Path | None = None
) -> tuple[dict[str, list[Record]], list[SplitLoad]]:
    """Like :func:`load_dataset`, and also return one :class:`SplitLoad` per split.

    The provenance (which backend served each split, and how many rows the cap removed) is
    meant for the audit report.
    """
    if not spec.source.split_map:
        raise ValueError("source.split_map is empty; name at least one split")
    if spec.max_rows_per_split is not None and spec.max_rows_per_split < 1:
        raise ValueError("max_rows_per_split must be at least 1 when set")

    if spec.drop_label_values and spec.label_field is None:
        raise ValueError("drop_label_values needs a label_field")

    records: dict[str, list[Record]] = {}
    loads: list[SplitLoad] = []
    for split, source_split in spec.source.split_map.items():
        source = _open_source(spec.source, source_split, base_dir)
        eligible = _eligible_positions(source, spec)
        n_eligible = source.n_rows if eligible is None else len(eligible)
        sampled_pos = _select_indices(n_eligible, spec.max_rows_per_split, spec.seed, split)
        if eligible is None:
            indices = sampled_pos
        elif sampled_pos is None:
            indices = eligible
        else:
            indices = [eligible[i] for i in sampled_pos]
        rows = source.take(indices)
        positions = range(source.n_rows) if indices is None else indices
        records[split] = [
            _to_record(row, index=idx, split=split, spec=spec)
            for idx, row in zip(positions, rows, strict=True)
        ]
        loads.append(
            SplitLoad(
                split=split,
                source_split=source_split,
                backend=source.backend,
                source_rows=source.n_rows,
                kept_rows=len(rows),
                sampled=sampled_pos is not None,
                dropped_by_label=source.n_rows - n_eligible,
            )
        )
    return records, loads


def _eligible_positions(source: _Source, spec: DatasetSpec) -> list[int] | None:
    """Source positions that survive ``drop_label_values``, or None when nothing is dropped.

    Label values are compared as stripped text, so ``-1`` matches the integer -1 and the
    string "-1". Dropping happens before the seeded cap, so the cap samples kept rows only.
    """
    if not spec.drop_label_values or spec.label_field is None:
        return None
    drop = {str(v).strip() for v in spec.drop_label_values}
    labels = source.column(spec.label_field)
    return [i for i, value in enumerate(labels) if _label_text(value) not in drop]


def _label_text(value: Any) -> str:
    """Text form of a label for comparison: None becomes "", otherwise ``str`` stripped."""
    return "" if value is None else str(value).strip()


def _split_seed(seed: int, split: str) -> int:
    """Derive a per-split seed from the dataset seed, stable across processes."""
    digest = hashlib.blake2b(f"{seed}\x1f{split}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big")


def _select_indices(n_rows: int, cap: int | None, seed: int, split: str) -> list[int] | None:
    """Return sorted source positions for a seeded sample of ``cap`` rows, or None for all.

    None means no sampling was needed (no cap, or the split is no larger than the cap).
    """
    if cap is None or n_rows <= cap:
        return None
    rng = np.random.default_rng(_split_seed(seed, split))
    chosen = rng.choice(n_rows, size=cap, replace=False)
    return sorted(int(i) for i in chosen)


def _to_record(row: Mapping[str, Any], *, index: int, split: str, spec: DatasetSpec) -> Record:
    """Turn one source row into a :class:`Record`, checking required columns."""
    for key in spec.text_fields:
        if key not in row:
            raise ValueError(
                f"split {split!r}: text field {key!r} is not a column (columns: {sorted(row)})"
            )
    if spec.label_field is not None and spec.label_field not in row:
        raise ValueError(f"split {split!r}: label field {spec.label_field!r} is not a column")
    if spec.id_field is not None and spec.id_field not in row:
        raise ValueError(f"split {split!r}: id field {spec.id_field!r} is not a column")

    excluded = {spec.label_field, spec.id_field}
    fields = {str(k): _as_text(v) for k, v in row.items() if k not in excluded}
    rec_id = _as_text(row[spec.id_field]) if spec.id_field is not None else f"{split}:{index}"
    label = _plain_label(row[spec.label_field]) if spec.label_field is not None else None
    return Record(id=rec_id, split=split, fields=fields, label=label)


def _as_text(value: Any) -> str:
    """Convert a cell value to text: None and NaN become "", containers become JSON."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, float) and math.isnan(value):
        return ""
    if isinstance(value, list | dict | tuple):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return str(value)


def _plain_label(value: Any) -> Any | None:
    """Return a JSON-friendly label: numpy scalars are unwrapped, empty values become None."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, str):
        return value if value != "" else None
    item = getattr(value, "item", None)
    if callable(item):
        value = item()
    return value


def _open_source(src: SourceSpec, source_split: str, base_dir: Path | None) -> _Source:
    """Open one source split. Dispatches on ``src.kind``."""
    if src.kind == "hf":
        return _open_hf(src, source_split)
    path = _local_path(src, source_split, base_dir)
    if src.kind == "jsonl":
        rows = _read_jsonl(path)
        return _Source(
            "jsonl",
            len(rows),
            lambda idx: _take_list(rows, idx),
            lambda name: [row.get(name) for row in rows],
        )
    if src.kind == "csv":
        rows_csv = _read_csv(path)
        return _Source(
            "csv",
            len(rows_csv),
            lambda idx: _take_list(rows_csv, idx),
            lambda name: [row.get(name) for row in rows_csv],
        )
    if src.kind == "parquet":
        import pyarrow.parquet as pq

        table = pq.read_table(path)
        return _Source(
            "parquet",
            table.num_rows,
            lambda idx: _take_table(table, idx),
            lambda name: table.column(name).to_pylist(),
        )
    raise ValueError(f"unsupported source kind: {src.kind!r}")


def _local_path(src: SourceSpec, source_split: str, base_dir: Path | None) -> Path:
    """Resolve the file for one split of a local source (see the module docstring)."""
    raw = src.id_or_path
    templated = "{split}" in raw
    path = Path(raw.replace("{split}", source_split))
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    if not templated and path.is_dir():
        path = path / f"{source_split}{_LOCAL_EXTENSIONS[src.kind]}"
    elif not templated and len(src.split_map) > 1:
        raise ValueError(
            f"{raw!r} is a single file but split_map names {len(src.split_map)} splits; "
            "use a directory or a path containing {split}"
        )
    if not path.is_file():
        raise FileNotFoundError(f"no {src.kind} file for split {source_split!r}: {path}")
    return path


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSON Lines file. Blank lines are skipped; every other line must be an object."""
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{lineno}: expected a JSON object per line")
            rows.append(obj)
    return rows


def _read_csv(path: Path) -> list[dict[str, Any]]:
    """Read a CSV file with a header row. A UTF-8 byte-order mark is accepted."""
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return [dict(row) for row in csv.DictReader(fh)]


def _take_list(rows: list[dict[str, Any]], indices: list[int] | None) -> list[dict[str, Any]]:
    """Return the rows at ``indices`` (all rows when None)."""
    if indices is None:
        return list(rows)
    return [rows[i] for i in indices]


def _take_table(table: Any, indices: list[int] | None) -> list[dict[str, Any]]:
    """Return the rows of a pyarrow table at ``indices`` (all rows when None)."""
    import pyarrow as pa

    if indices is None:
        return list(table.to_pylist())
    return list(table.take(pa.array(indices, type=pa.int64())).to_pylist())


def _open_hf(src: SourceSpec, source_split: str) -> _Source:
    """Open a Hugging Face split with ``datasets``, falling back to the parquet branch."""
    try:
        import datasets

        ds = datasets.load_dataset(
            src.id_or_path, name=src.config, split=source_split, revision=src.revision
        )
    except Exception as exc:
        logger.warning(
            "datasets.load_dataset failed for %s (%s); trying parquet fallback",
            src.id_or_path,
            source_split,
        )
        return _open_hf_parquet(src, source_split, reason=repr(exc))
    return _Source(
        "hf-datasets",
        len(ds),
        lambda idx: _take_hf(ds, idx),
        lambda name: list(ds[name]),
    )


def _take_hf(ds: Any, indices: list[int] | None) -> list[dict[str, Any]]:
    """Return rows of a ``datasets.Dataset`` at ``indices`` (all rows when None)."""
    if indices is None:
        return list(ds.to_list())
    return list(ds.select(indices).to_list())


def _open_hf_parquet(src: SourceSpec, source_split: str, *, reason: str) -> _Source:
    """Read a split from the parquet conversion branch of a Hugging Face dataset."""
    import huggingface_hub
    import pyarrow as pa
    import pyarrow.parquet as pq

    config = src.config or DEFAULT_CONFIG
    files = huggingface_hub.list_repo_files(
        src.id_or_path, repo_type="dataset", revision=PARQUET_REVISION
    )
    wanted = sorted(f for f in files if _is_split_file(f, config, source_split))
    if not wanted:
        raise FileNotFoundError(
            f"no parquet files for {config}/{source_split} on {PARQUET_REVISION} of "
            f"{src.id_or_path} (datasets error: {reason})"
        )
    tables = []
    for name in wanted:
        local = huggingface_hub.hf_hub_download(
            src.id_or_path, name, repo_type="dataset", revision=PARQUET_REVISION
        )
        tables.append(pq.read_table(local))
    table = pa.concat_tables(tables)
    return _Source(
        "hf-parquet-fallback",
        table.num_rows,
        lambda idx: _take_table(table, idx),
        lambda name: table.column(name).to_pylist(),
    )


def _is_split_file(path: str, config: str, split: str) -> bool:
    """True for ``<config>/<split>/<shard>.parquet`` paths on the convert branch."""
    parts = path.split("/")
    return (
        len(parts) == 3
        and parts[0] == config
        and parts[1] == split
        and parts[2].endswith(".parquet")
    )
