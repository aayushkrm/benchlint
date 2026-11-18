"""Decontamination: 13-gram overlap between eval items and a reference corpus."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from benchlint.schema import Finding, NgramConfig, Record


def find_ngram_overlap(
    eval_records: Sequence[Record],
    reference_texts: Iterable[str],
    text_fields: Sequence[str],
    config: NgramConfig,
) -> Finding:
    """Count eval items sharing at least one ``config.n``-gram with the reference texts.

    ``reference_texts`` is streamed; the caller loads it from ``config.reference``.
    """
    raise NotImplementedError
