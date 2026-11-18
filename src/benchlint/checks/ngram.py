"""Decontamination: 13-gram overlap between eval items and a reference corpus.

Each eval item is scored by the fraction of its distinct word n-grams that also occur in the
reference corpus. An n-gram is hashed (blake2b, 64 bits, deterministic across processes), so
the index stores integers and not strings. Text is normalised with the same tokeniser as
``near.py``. Items with fewer than ``n`` tokens have no n-grams and are not evaluable.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from typing import Any

from benchlint.checks.near import _record_text, _tokens
from benchlint.schema import ExamplePair, Finding, NgramConfig, Record

_EXAMPLE_LIMIT = 5
_TEXT_LIMIT = 200
_HIST_BINS = 10
_THRESHOLDS: tuple[float, ...] = (0.5, 0.8, 1.0)


def _hash_ngram(gram: Sequence[str]) -> int:
    """Return a deterministic 64-bit hash of a token n-gram."""
    digest = hashlib.blake2b(" ".join(gram).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big")


def _ngram_hashes(tokens: Sequence[str], n: int) -> set[int]:
    """Return the set of hashed n-grams of ``tokens``; empty if there are fewer than ``n``."""
    if len(tokens) < n:
        return set()
    return {_hash_ngram(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def _snippet(text: str) -> str:
    """Return the first ``_TEXT_LIMIT`` characters of ``text``."""
    return text[:_TEXT_LIMIT]


def find_ngram_overlap(
    eval_records: Sequence[Record],
    reference_texts: Iterable[str],
    text_fields: Sequence[str],
    config: NgramConfig,
) -> Finding:
    """Count eval items sharing at least one ``config.n``-gram with the reference texts.

    The reference is streamed once and indexed by the hash of each distinct n-gram, keeping
    the index of the first reference text that contains it. Each eval item gets a
    contamination fraction: distinct n-grams found in the reference divided by distinct
    n-grams of the item. The Finding's ``count`` is the number of items with fraction above
    zero, and ``rate`` is ``count / denominator``, where the denominator is the number of
    evaluable items (at least ``n`` tokens). ``details`` holds threshold counts, a histogram
    of fractions and the reference size. ``examples`` are the (at most 5) items with the
    highest fraction, with the matched reference text cut to 200 characters.

    Memory grows with the number of distinct reference n-grams, and the snippets of all
    reference texts are kept. Cap the reference size for large corpora.

    Args:
        eval_records: Items to check. Each must contain every field in ``text_fields``.
        reference_texts: Reference documents as strings. Consumed once, in order.
        text_fields: Field names whose values are concatenated, in order, into the text.
        config: n-gram size (``n``); ``config.reference`` is loaded by the caller.

    Returns:
        A single ``Finding`` with ``check="ngram"``.

    Raises:
        ValueError: If ``config.n`` is below 1, or a record lacks one of ``text_fields``.
    """
    n = config.n
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")

    ref_index: dict[int, int] = {}
    ref_snippets: list[str] = []
    for ridx, text in enumerate(reference_texts):
        ref_snippets.append(_snippet(text))
        for h in _ngram_hashes(_tokens(text), n):
            ref_index.setdefault(h, ridx)

    fractions: list[float] = []
    flagged: list[tuple[float, int, int]] = []  # (fraction, eval index, first ref index)
    for idx, record in enumerate(eval_records):
        grams = _ngram_hashes(_tokens(_record_text(record, text_fields)), n)
        if not grams:
            continue
        matched = [h for h in grams if h in ref_index]
        fraction = len(matched) / len(grams)
        fractions.append(fraction)
        if matched:
            flagged.append((fraction, idx, min(ref_index[h] for h in matched)))

    denominator = len(fractions)
    count = len(flagged)
    ranked = sorted(flagged, key=lambda t: (-t[0], eval_records[t[1]].id))
    examples = [
        ExamplePair(
            a_id=eval_records[idx].id,
            b_id=f"reference:{ridx}",
            a_split=eval_records[idx].split,
            b_split="reference",
            similarity=fraction,
            a_text=_record_text(eval_records[idx], text_fields)[:_TEXT_LIMIT],
            b_text=ref_snippets[ridx],
        )
        for fraction, idx, ridx in ranked[:_EXAMPLE_LIMIT]
    ]

    counts_at: dict[str, int] = {"any": count}
    for threshold in _THRESHOLDS:
        counts_at[f"ge_{threshold}"] = sum(1 for f in fractions if f >= threshold)
    histogram = [0] * _HIST_BINS
    for fraction in fractions:
        histogram[min(int(fraction * _HIST_BINS), _HIST_BINS - 1)] += 1

    details: dict[str, Any] = {
        "n": n,
        "n_eval_items": len(eval_records),
        "n_items_evaluable": denominator,
        "n_items_too_short": len(eval_records) - denominator,
        "n_reference_texts": len(ref_snippets),
        "n_reference_ngrams": len(ref_index),
        "counts_at_fraction": counts_at,
        "fraction_histogram": {
            "bin_edges": [round(i / _HIST_BINS, 2) for i in range(_HIST_BINS + 1)],
            "counts": histogram,
        },
        "mean_fraction": sum(fractions) / denominator if denominator else 0.0,
    }
    return Finding(
        check="ngram",
        severity="info" if count == 0 else "medium",
        splits=sorted({r.split for r in eval_records}),
        count=count,
        denominator=denominator,
        rate=count / denominator if denominator else 0.0,
        examples=examples,
        details=details,
    )
