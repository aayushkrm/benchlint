"""Near-duplicate detection with MinHash LSH (datasketch).

Word shingles of size ``config.shingle``, ``config.num_perm`` permutations and a Jaccard
threshold of ``config.threshold``. Candidate pairs from LSH are verified with exact Jaccard
on the shingle sets, so every reported pair really has Jaccard at or above the threshold.
LSH recall is probabilistic: a pair just above the threshold can be missed. The seed fixes
the permutations, so a run is reproducible.

Text normalisation: ``benchlint.normalize`` is owned by another work package and may not be
ready. This module therefore uses its own private ``_norm`` (NFKC, casefold, punctuation to
spaces, whitespace collapse). It is the default and the only normaliser used here. Once
``normalize_text`` is implemented, the two should be unified before any result is produced.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Sequence
from typing import Any

from datasketch import MinHash, MinHashLSH

from benchlint.schema import ExamplePair, Finding, NearConfig, Record

_EXAMPLE_LIMIT = 5
_TEXT_LIMIT = 200
_HIST_BINS = 10


def _norm(text: str) -> str:
    """Return the normalised form of ``text``: NFKC, casefold, punctuation to spaces.

    Punctuation and symbol characters become spaces (so "don't" becomes "don t"), and runs
    of whitespace collapse to one space.
    """
    folded = unicodedata.normalize("NFKC", text).casefold()
    chars = [" " if unicodedata.category(ch)[0] in "PS" else ch for ch in folded]
    return " ".join("".join(chars).split())


def _tokens(text: str) -> list[str]:
    """Return the whitespace tokens of the normalised ``text``."""
    return _norm(text).split()


def _record_text(record: Record, text_fields: Sequence[str]) -> str:
    """Join the named fields of ``record`` with a space, in the given order.

    A missing field is an error, not a silent empty string, so a malformed dataset cannot
    quietly lose its text.
    """
    missing = [name for name in text_fields if name not in record.fields]
    if missing:
        raise ValueError(f"record {record.id!r} in split {record.split!r} lacks fields {missing}")
    return " ".join(record.fields[name] for name in text_fields)


def shingle_set(tokens: Sequence[str], k: int) -> frozenset[str]:
    """Return the set of word ``k``-gram shingles of ``tokens``.

    A text shorter than ``k`` tokens gives one shingle, the whole text. An empty token list
    gives the empty set, and such records are excluded from matching.
    """
    if not tokens:
        return frozenset()
    if len(tokens) < k:
        return frozenset({" ".join(tokens)})
    return frozenset(" ".join(tokens[i : i + k]) for i in range(len(tokens) - k + 1))


def _minhash(shingles: frozenset[str], num_perm: int, seed: int) -> MinHash:
    """Return the MinHash signature of a shingle set (independent of set iteration order)."""
    sig = MinHash(num_perm=num_perm, seed=seed)
    for shingle in sorted(shingles):
        sig.update(shingle.encode("utf-8"))
    return sig


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    """Return the exact Jaccard similarity of two non-empty shingle sets."""
    return len(a & b) / len(a | b)


def _histogram(values: Sequence[float]) -> dict[str, list[Any]]:
    """Return ``_HIST_BINS`` equal-width bins on [0, 1]; the value 1.0 falls in the last bin."""
    counts = [0] * _HIST_BINS
    for value in values:
        counts[min(int(value * _HIST_BINS), _HIST_BINS - 1)] += 1
    edges = [round(i / _HIST_BINS, 2) for i in range(_HIST_BINS + 1)]
    return {"bin_edges": edges, "counts": counts}


class _Scope:
    """Accumulates candidate and near-duplicate pairs for one scope (cross or within split)."""

    def __init__(self) -> None:
        """Start with no pairs."""
        self.candidates: list[float] = []
        self.near: list[tuple[float, int, int]] = []

    def add(self, sim: float, a: int, b: int, threshold: float) -> None:
        """Record one verified candidate pair; keep it as near if it clears ``threshold``."""
        self.candidates.append(sim)
        if sim >= threshold:
            self.near.append((sim, a, b))


def _finding(
    scope_name: str,
    scope: _Scope,
    records: Sequence[Record],
    text_fields: Sequence[str],
    splits: list[str],
    denominator: int,
    config: NearConfig,
    seed: int,
) -> Finding:
    """Build one near-duplicate ``Finding`` from a scope accumulator.

    ``count`` is the number of distinct items with at least one near-duplicate partner in the
    scope. ``denominator`` is the number of evaluable records (non-empty text).
    """
    flagged: set[int] = set()
    flagged_by_split: dict[str, int] = dict.fromkeys(splits, 0)
    pairs_by_split_pair: dict[str, int] = {}
    for _, a, b in scope.near:
        flagged.update((a, b))
        sa, sb = records[a].split, records[b].split
        key = f"{min(sa, sb)}|{max(sa, sb)}"
        pairs_by_split_pair[key] = pairs_by_split_pair.get(key, 0) + 1
    for idx in flagged:
        flagged_by_split[records[idx].split] += 1

    ranked = sorted(scope.near, key=lambda t: (-t[0], records[t[1]].id, records[t[2]].id))
    examples = [
        ExamplePair(
            a_id=records[a].id,
            b_id=records[b].id,
            a_split=records[a].split,
            b_split=records[b].split,
            similarity=sim,
            a_text=_record_text(records[a], text_fields)[:_TEXT_LIMIT],
            b_text=_record_text(records[b], text_fields)[:_TEXT_LIMIT],
        )
        for sim, a, b in ranked[:_EXAMPLE_LIMIT]
    ]

    count = len(flagged)
    details: dict[str, Any] = {
        "scope": scope_name,
        "threshold": config.threshold,
        "num_perm": config.num_perm,
        "shingle": config.shingle,
        "seed": seed,
        "n_records": len(records),
        "n_evaluable": denominator,
        "n_candidate_pairs": len(scope.candidates),
        "n_near_pairs": len(scope.near),
        "n_candidate_pairs_below_threshold": sum(
            1 for s in scope.candidates if s < config.threshold
        ),
        "jaccard_histogram": _histogram(scope.candidates),
        "flagged_by_split": flagged_by_split,
        "near_pairs_by_split_pair": pairs_by_split_pair,
        "note": (
            "rate is flagged items / evaluable items; the Jaccard histogram covers LSH candidate "
            "pairs of this scope, each verified by exact Jaccard"
        ),
    }
    return Finding(
        check="near",
        severity="info" if count == 0 else "medium",
        splits=splits,
        count=count,
        denominator=denominator,
        rate=count / denominator if denominator else 0.0,
        examples=examples,
        details=details,
    )


def find_near_duplicates(
    records: Sequence[Record],
    text_fields: Sequence[str],
    config: NearConfig,
    *,
    seed: int = 0,
) -> list[Finding]:
    """Return near-duplicate findings, with the Jaccard distribution in ``details``.

    Returns up to two findings, both with ``check="near"``. The ``cross_split`` finding is
    present only when at least two splits exist, and counts items with a near-duplicate in a
    different split. The ``within_split`` finding counts items with one in their own split.
    ``rate`` is ``count / denominator``, where the denominator is the number of evaluable
    records (non-empty text). Examples are the highest-Jaccard pairs (at most 5), with texts
    cut to 200 characters. The Jaccard histogram covers each scope's LSH candidate pairs,
    which exposes template-driven false positives (for example GSM8K phrasing).

    Args:
        records: Rows to check. Each row must contain every field in ``text_fields``.
        text_fields: Field names whose values are concatenated, in order, into the text.
        config: Shingle size, permutations and Jaccard threshold.
        seed: Seed for the MinHash permutations. The same seed gives the same output.

    Returns:
        Findings with ``check="near"``, cross-split first when present. Empty ``records``
        give an empty list.

    Raises:
        ValueError: If a record lacks one of ``text_fields``.
    """
    if not records:
        return []
    splits = sorted({record.split for record in records})

    shingles: dict[int, frozenset[str]] = {}
    signatures: dict[int, MinHash] = {}
    lsh = MinHashLSH(threshold=config.threshold, num_perm=config.num_perm)
    for idx, record in enumerate(records):
        sh = shingle_set(_tokens(_record_text(record, text_fields)), config.shingle)
        if not sh:
            continue
        shingles[idx] = sh
        signatures[idx] = _minhash(sh, config.num_perm, seed)
        lsh.insert(str(idx), signatures[idx])

    candidate_pairs: set[tuple[int, int]] = set()
    for idx, sig in signatures.items():
        for key in lsh.query(sig):
            other = int(key)
            if other > idx:
                candidate_pairs.add((idx, other))

    cross, within = _Scope(), _Scope()
    for a, b in sorted(candidate_pairs):
        sim = _jaccard(shingles[a], shingles[b])
        target = cross if records[a].split != records[b].split else within
        target.add(sim, a, b, config.threshold)

    evaluable = len(shingles)
    findings: list[Finding] = []
    if len(splits) >= 2:
        findings.append(
            _finding("cross_split", cross, records, text_fields, splits, evaluable, config, seed)
        )
    findings.append(
        _finding("within_split", within, records, text_fields, splits, evaluable, config, seed)
    )
    return findings
