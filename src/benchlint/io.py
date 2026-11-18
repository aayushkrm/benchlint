"""Dataset loaders: Hugging Face (with a parquet fallback), JSONL, CSV and parquet.

Yields ``Record(id, split, fields, label)``. ``max_rows_per_split`` caps each split with a
seeded sample, so runs are reproducible.
"""

from __future__ import annotations

from pathlib import Path

from benchlint.schema import DatasetSpec, Record


def load_dataset(spec: DatasetSpec, *, base_dir: Path | None = None) -> dict[str, list[Record]]:
    """Load every split named in ``spec.source.split_map`` and return them keyed by split name.

    ``base_dir`` is used to resolve relative paths for local (jsonl, csv, parquet) sources;
    the audit CLI passes the directory that holds the spec file.
    Hugging Face sources try ``datasets`` first and fall back to parquet files from the
    ``refs/convert/parquet`` revision via ``huggingface_hub``.
    """
    raise NotImplementedError
