"""Label conflicts: duplicate or near-duplicate items that carry different labels.

Only records with a label (not None and not blank) and a non-empty key are considered. A
group is a set of records that are exact duplicates (same key), or in near mode, a connected
component of near-duplicates. A group is in conflict when its members have more than one
distinct label (compared as ``str(label).strip()``).

Finding conventions:

- ``count`` is the number of records that belong to a conflicting group; ``denominator`` is
  the number of labelled records with a non-empty key.
- Severity: ``info`` when there is no conflict, ``high`` when ``rate >= 0.01``, else
  ``medium``.
- Near mode uses word 5-gram shingles (or the whole text when shorter), MinHash LSH with
  ``NearConfig.num_perm`` and a fixed seed, candidates checked by exact Jaccard against
  ``NearConfig.threshold``. Components are single-linkage, so chains of pairs can join.
  Exact-duplicate groups are always joined, whatever their shingles.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from typing import Any, Final

from benchlint.checks.exact import (
    EXAMPLE_LIMIT,
    HIGH_RATE,
    PAIR_LIMIT,
    Severity,
    exact_keys,
    example_pair,
)
from benchlint.checks.near import jaccard, lsh_candidates, record_shingles
from benchlint.schema import ExamplePair, Finding, NearConfig, Record

LABEL_TOP_K: Final = 20
"""Number of label values kept in ``details["label_counts_in_conflicts"]``."""


def find_label_conflicts(
    records: Sequence[Record],
    text_fields: Sequence[str],
    *,
    near: NearConfig | None = None,
    mask_numbers: bool = False,
    seed: int = 0,
) -> list[Finding]:
    """Find groups of duplicates with more than one distinct label.

    If ``near`` is None only exact duplicates are grouped; otherwise near-duplicates too
    (see the module docstring). ``seed`` fixes the MinHash permutations in near mode. Returns
    one ``Finding`` (check ``"labels"``), or ``[]`` when no record has a label and a
    non-empty key.
    """
    keys = exact_keys(records, text_fields, mask_numbers=mask_numbers)
    items = [i for i, r in enumerate(records) if _has_label(r.label) and keys[i] is not None]
    if not items:
        return []

    shingles: dict[int, frozenset[str]] = {}
    if near is None:
        groups = _exact_groups(items, keys)
        mode = "exact"
    else:
        shingles = {
            i: record_shingles(records[i], text_fields, near.shingle, mask_numbers=mask_numbers)
            for i in items
        }
        groups = _near_groups(items, keys, shingles, near, seed)
        mode = "near"

    conflicts: list[list[int]] = []
    label_counts: Counter[str] = Counter()
    n_checked = 0
    for group in groups:
        if len(group) < 2:
            continue
        n_checked += 1
        texts = {_label_text(records[i].label) for i in group}
        if len(texts) > 1:
            conflicts.append(group)
            label_counts.update(_label_text(records[i].label) for i in group)

    count = sum(len(g) for g in conflicts)
    denominator = len(items)
    rate = count / denominator
    severity: Severity
    if count == 0:
        severity = "info"
    elif rate >= HIGH_RATE:
        severity = "high"
    else:
        severity = "medium"

    pairs: list[list[Any]] = []
    pairs_total = 0
    for group in conflicts:
        for x in range(len(group)):
            for y in range(x + 1, len(group)):
                a, b = group[x], group[y]
                if _label_text(records[a].label) == _label_text(records[b].label):
                    continue
                pairs_total += 1
                if len(pairs) < PAIR_LIMIT:
                    sim = 1.0 if mode == "exact" else jaccard(shingles[a], shingles[b])
                    pairs.append([records[a].id, records[b].id, round(sim, 6)])

    examples: list[ExamplePair] = []
    for group in conflicts[:EXAMPLE_LIMIT]:
        a = group[0]
        a_label = _label_text(records[a].label)
        b = next(i for i in group if _label_text(records[i].label) != a_label)
        similarity = 1.0 if mode == "exact" else jaccard(shingles[a], shingles[b])
        examples.append(example_pair(records[a], records[b], text_fields, similarity))

    split_order = list(dict.fromkeys(r.split for r in records))
    in_conflict = {records[i].split for g in conflicts for i in g}
    splits = (
        [s for s in split_order if s in in_conflict]
        if conflicts
        else [s for s in split_order if any(records[i].split == s for i in items)]
    )
    return [
        Finding(
            check="labels",
            severity=severity,
            splits=splits,
            count=count,
            denominator=denominator,
            rate=rate,
            examples=examples,
            details={
                "mode": mode,
                "n_groups_checked": n_checked,
                "n_conflict_groups": len(conflicts),
                "pairs": pairs,
                "pairs_listed": len(pairs),
                "pairs_truncated": pairs_total > len(pairs),
                "label_counts_in_conflicts": dict(label_counts.most_common(LABEL_TOP_K)),
                "seed": seed,
                "near": None
                if near is None
                else {
                    "threshold": near.threshold,
                    "num_perm": near.num_perm,
                    "shingle": near.shingle,
                },
            },
        )
    ]


def _has_label(label: object) -> bool:
    """True when a label is present: not None and not blank after stripping."""
    return label is not None and str(label).strip() != ""


def _label_text(label: object) -> str:
    """Return the comparison form of a label: ``str(label)`` with surrounding space removed."""
    return str(label).strip()


def _exact_groups(items: Sequence[int], keys: Sequence[str | None]) -> list[list[int]]:
    """Group record indices by exact key (groups of any size, in first-seen order)."""
    by_key: dict[str, list[int]] = defaultdict(list)
    for i in items:
        key = keys[i]
        if key is not None:
            by_key[key].append(i)
    return list(by_key.values())


def _near_groups(
    items: Sequence[int],
    keys: Sequence[str | None],
    shingles: dict[int, frozenset[str]],
    near: NearConfig,
    seed: int,
) -> list[list[int]]:
    """Return connected components of exact and near-duplicate records (size >= 2 only).

    Candidate pairs come from the shared :func:`benchlint.checks.near.lsh_candidates`, which
    returns exact Jaccard; a pair joins a component when it reaches ``near.threshold``.
    """
    m = len(items)
    parent = list(range(m))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[max(rx, ry)] = min(rx, ry)

    first_pos: dict[str, int] = {}
    for pos, i in enumerate(items):
        key = keys[i]
        if key is None:
            continue
        if key in first_pos:
            union(pos, first_pos[key])
        else:
            first_pos[key] = pos

    positional = {pos: shingles[i] for pos, i in enumerate(items)}
    for (p, q), sim in lsh_candidates(positional, near, seed).items():
        if sim >= near.threshold:
            union(p, q)

    components: dict[int, list[int]] = defaultdict(list)
    for pos, i in enumerate(items):
        components[find(pos)].append(i)
    return [members for members in components.values() if len(members) >= 2]
