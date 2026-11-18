"""Near-duplicate detection with MinHash LSH (datasketch).

Word shingles of size ``config.shingle``, ``config.num_perm`` permutations and a Jaccard
threshold of ``config.threshold``. Candidate pairs come from LSH with the retrieval threshold
``min(config.threshold, config.lsh_threshold)``; every candidate is then checked with exact
Jaccard on the shingle sets. A pair is reported as near-duplicate only if its exact Jaccard is
at or above ``config.threshold``, so reported pairs are never below the reporting threshold.
LSH recall is probabilistic: a pair just above the threshold can still be missed. The seed
fixes the permutations, so a run is reproducible.

Text tokens are :func:`benchlint.normalize.word_tokens` (applied per field). The
helpers below (``record_shingles``, ``minhash_of``, ``jaccard``, ``lsh_candidates``) are the
single implementation of MinHash and LSH in benchlint; ``checks/labels.py`` uses them too.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from datasketch import MinHash, MinHashLSH

from benchlint.checks.exact import PAIR_LIMIT
from benchlint.normalize import word_tokens
from benchlint.schema import ExamplePair, Finding, NearConfig, Record

_EXAMPLE_LIMIT = 5
_TEXT_LIMIT = 200
_HIST_BINS = 10


def _tokens(text: str) -> list[str]:
    """Word tokens of ``text`` (see :func:`benchlint.normalize.word_tokens`)."""
    return word_tokens(text)


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


def record_shingles(
    record: Record,
    text_fields: Sequence[str],
    k: int,
    *,
    mask_numbers: bool = False,
) -> frozenset[str]:
    """Word ``k``-gram shingles of a record's text, each field normalised on its own.

    Fields are normalised separately so a boundary between two fields never merges words.
    Raises ``ValueError`` if a field is missing.
    """
    missing = [name for name in text_fields if name not in record.fields]
    if missing:
        raise ValueError(f"record {record.id!r} in split {record.split!r} lacks fields {missing}")
    words: list[str] = []
    for name in text_fields:
        words.extend(word_tokens(record.fields[name], mask_numbers=mask_numbers))
    return shingle_set(words, k)


def minhash_of(shingles: frozenset[str], num_perm: int, seed: int) -> MinHash:
    """MinHash signature of a shingle set. Independent of set iteration order."""
    sig = MinHash(num_perm=num_perm, seed=seed)
    for shingle in sorted(shingles):
        sig.update(shingle.encode("utf-8"))
    return sig


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    """Exact Jaccard similarity of two shingle sets (0.0 when both are empty)."""
    union = len(a | b)
    if union == 0:
        return 0.0
    return len(a & b) / union


def lsh_retrieval_threshold(config: NearConfig) -> float:
    """The LSH retrieval threshold actually used: ``min(threshold, lsh_threshold)``."""
    return min(config.threshold, config.lsh_threshold)


def lsh_candidates(
    shingles: Mapping[int, frozenset[str]],
    config: NearConfig,
    seed: int,
) -> dict[tuple[int, int], float]:
    """Candidate pairs from MinHash LSH, each with its exact Jaccard.

    Keys are index pairs ``(a, b)`` with ``a < b`` taken from ``shingles``. Empty shingle sets
    are never candidates. The returned Jaccard is exact, so callers only apply their own
    threshold.
    """
    lsh = MinHashLSH(threshold=lsh_retrieval_threshold(config), num_perm=config.num_perm)
    signatures: dict[int, MinHash] = {}
    for idx in sorted(shingles):
        sh = shingles[idx]
        if not sh:
            continue
        signatures[idx] = minhash_of(sh, config.num_perm, seed)
        lsh.insert(str(idx), signatures[idx])

    pairs: dict[tuple[int, int], float] = {}
    for idx, sig in signatures.items():
        for key in lsh.query(sig):
            other = int(key)
            if other > idx:
                pairs[(idx, other)] = jaccard(shingles[idx], shingles[other])
    return pairs


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
    listed = ranked[:PAIR_LIMIT]
    pairs = [[records[a].id, records[b].id, round(sim, 6)] for sim, a, b in listed]

    count = len(flagged)
    details: dict[str, Any] = {
        "scope": scope_name,
        "threshold": config.threshold,
        "lsh_threshold": lsh_retrieval_threshold(config),
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
        "pairs": pairs,
        "pairs_listed": len(pairs),
        "pairs_truncated": len(scope.near) > len(pairs),
        "note": (
            "rate is flagged items / evaluable items; the Jaccard histogram covers LSH candidate "
            "pairs of this scope, each verified by exact Jaccard; pairs lists near-duplicate "
            "pairs [a_id, b_id, jaccard] with jaccard at or above the threshold"
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
    """Return near-duplicate findings, with the Jaccard distribution and pairs in ``details``.

    Returns up to two findings, both with ``check="near"``. The ``cross_split`` finding is
    present only when at least two splits exist, and counts items with a near-duplicate in a
    different split. The ``within_split`` finding counts items with one in their own split.
    ``rate`` is ``count / denominator``, where the denominator is the number of evaluable
    records (non-empty text). Examples are the highest-Jaccard pairs (at most 5), with texts
    cut to 200 characters. ``details["pairs"]`` lists near-duplicate pairs as
    ``[a_id, b_id, jaccard]`` (at most ``PAIR_LIMIT``). The Jaccard histogram covers each
    scope's LSH candidate pairs, which exposes template-driven false positives.

    Args:
        records: Rows to check. Each row must contain every field in ``text_fields``.
        text_fields: Field names whose values are normalised and joined into the text.
        config: Shingle size, permutations, reporting and LSH thresholds.
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
    for idx, record in enumerate(records):
        sh = record_shingles(record, text_fields, config.shingle)
        if sh:
            shingles[idx] = sh

    cross, within = _Scope(), _Scope()
    for (a, b), sim in sorted(lsh_candidates(shingles, config, seed).items()):
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
