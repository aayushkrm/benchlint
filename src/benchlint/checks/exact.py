"""Exact duplicate detection: within each split and across splits.

Key is the blake2b digest of the normalised, concatenated text fields. Each field is
normalised on its own and the normalised parts are joined with the unit separator, so field
boundaries survive (see :mod:`benchlint.normalize`).

Finding conventions:

- Within a split: ``count`` is the number of redundant copies (records that repeat an
  earlier record's key); ``denominator`` is the number of records in that split with a
  non-empty key. Records whose fields all normalise to empty are skipped and counted in
  ``details["skipped_empty"]``.
- Across two splits ``A`` and ``B``, where ``B`` is the later of the two in the order the
  splits first appear in ``records`` (the spec's ``split_map`` order): ``count`` is the
  number of ``B`` records whose key also occurs in ``A``; ``denominator`` is the number of
  ``B`` records with a non-empty key. The reverse direction is in ``details``.
- Severity: a within-split duplicate is ``low`` when it exists and ``info`` otherwise. A
  cross-split duplicate is ``high`` when ``rate >= 0.01``, ``medium`` when ``count > 0`` and
  ``info`` otherwise. Those thresholds are documented here and apply to every audit.
- A finding is emitted for every checked split and every split pair, even when the count is
  zero (``info``), so a report shows what was checked.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any, Final, Literal

from benchlint.normalize import FIELD_SEPARATOR, normalize_fields
from benchlint.schema import ExamplePair, Finding, Record

EXAMPLE_LIMIT: Final = 5
"""Maximum number of ``ExamplePair`` objects per finding."""
PAIR_LIMIT: Final = 5000
"""Most duplicate pairs listed per finding in ``details["pairs"]``."""
TEXT_LIMIT: Final = 200
"""Characters kept from each example text."""
DIGEST_SIZE: Final = 16
"""blake2b digest size in bytes (128 bits)."""
HIGH_RATE: Final = 0.01
"""Cross-split duplicate rate at or above which severity is ``high``."""
Severity = Literal["info", "low", "medium", "high"]
"""Allowed values of ``Finding.severity``."""
KEY_METHOD: Final = "blake2b-128 over per-field NFKC/casefold/punct-strip/whitespace-collapse"


def exact_key(
    fields: Mapping[str, str], text_fields: Sequence[str], *, mask_numbers: bool = False
) -> str | None:
    """Return the exact-duplicate key for one record, or None if its fields normalise to empty.

    The key is a blake2b hex digest of the normalised field values joined with the unit
    separator.
    """
    parts = normalize_fields(fields, text_fields, mask_numbers=mask_numbers)
    if not any(parts):
        return None
    payload = FIELD_SEPARATOR.join(parts).encode("utf-8")
    return hashlib.blake2b(payload, digest_size=DIGEST_SIZE).hexdigest()


def exact_keys(
    records: Sequence[Record], text_fields: Sequence[str], *, mask_numbers: bool = False
) -> list[str | None]:
    """Return one key per record (aligned with ``records``); None marks an empty key."""
    return [exact_key(r.fields, text_fields, mask_numbers=mask_numbers) for r in records]


def snippet(record: Record, text_fields: Sequence[str]) -> str:
    """Return the raw text of a record's text fields for display, cut to ``TEXT_LIMIT``."""
    text = " | ".join(record.fields[k] for k in text_fields if k in record.fields)
    return text[:TEXT_LIMIT]


def example_pair(
    a: Record, b: Record, text_fields: Sequence[str], similarity: float
) -> ExamplePair:
    """Build an :class:`ExamplePair` from two records with texts truncated to 200 characters."""
    return ExamplePair(
        a_id=a.id,
        b_id=b.id,
        a_split=a.split,
        b_split=b.split,
        similarity=similarity,
        a_text=snippet(a, text_fields),
        b_text=snippet(b, text_fields),
    )


def _rate(count: int, denominator: int) -> float:
    """Return ``count / denominator``, or 0.0 when the denominator is zero."""
    return count / denominator if denominator else 0.0


def find_exact_duplicates(
    records: Sequence[Record],
    text_fields: Sequence[str],
    *,
    mask_numbers: bool = False,
) -> list[Finding]:
    """Return one ``Finding`` per split (within-split duplicates) and per split pair.

    Splits are taken in the order they first appear in ``records``. Each finding lists its
    ``splits``, the duplicate ``count``, the ``denominator`` and up to 5 example pairs (see the
    module docstring for the exact definitions and severity rule). Empty input returns ``[]``.
    """
    keys = exact_keys(records, text_fields, mask_numbers=mask_numbers)
    splits = list(dict.fromkeys(r.split for r in records))
    keyed: dict[str, list[tuple[str, Record]]] = {s: [] for s in splits}
    skipped: dict[str, int] = {s: 0 for s in splits}
    for rec, key in zip(records, keys, strict=True):
        if key is None:
            skipped[rec.split] += 1
        else:
            keyed[rec.split].append((key, rec))

    findings: list[Finding] = []
    for split in splits:
        findings.append(
            _within_split(
                split,
                keyed[split],
                skipped[split],
                text_fields,
                mask_numbers=mask_numbers,
            )
        )
    for i, first in enumerate(splits):
        for second in splits[i + 1 :]:
            findings.append(
                _across_splits(
                    first, second, keyed[first], keyed[second], text_fields, mask_numbers
                )
            )
    return findings


def _within_split(
    split: str,
    items: Sequence[tuple[str, Record]],
    skipped: int,
    text_fields: Sequence[str],
    *,
    mask_numbers: bool,
) -> Finding:
    """Build the within-split finding for one split."""
    groups: dict[str, list[Record]] = defaultdict(list)
    for key, rec in items:
        groups[key].append(rec)

    redundant = 0
    n_groups = 0
    largest = 0
    examples: list[ExamplePair] = []
    pairs: list[list[Any]] = []
    for members in groups.values():
        if len(members) < 2:
            continue
        n_groups += 1
        largest = max(largest, len(members))
        redundant += len(members) - 1
        first = members[0]
        for dup in members[1:]:
            if len(examples) < EXAMPLE_LIMIT:
                examples.append(example_pair(first, dup, text_fields, 1.0))
            if len(pairs) < PAIR_LIMIT:
                pairs.append([first.id, dup.id, 1.0])

    denominator = len(items)
    return Finding(
        check="exact",
        severity="low" if redundant > 0 else "info",
        splits=[split],
        count=redundant,
        denominator=denominator,
        rate=_rate(redundant, denominator),
        examples=examples,
        details={
            "scope": "within",
            "pairs": pairs,
            "pairs_truncated": redundant > len(pairs),
            "n_groups": n_groups,
            "largest_group": largest,
            "skipped_empty": skipped,
            "mask_numbers": mask_numbers,
            "key_method": KEY_METHOD,
        },
    )


def _across_splits(
    first: str,
    second: str,
    items_first: Sequence[tuple[str, Record]],
    items_second: Sequence[tuple[str, Record]],
    text_fields: Sequence[str],
    mask_numbers: bool,
) -> Finding:
    """Build the cross-split finding for the pair ``(first, second)``.

    The measured split is ``second``; ``count`` is the number of its records whose key also
    occurs in ``first``.
    """
    first_index: dict[str, Record] = {}
    for key, rec in items_first:
        first_index.setdefault(key, rec)
    second_keys = {key for key, _ in items_second}

    count = 0
    examples: list[ExamplePair] = []
    pairs: list[list[Any]] = []
    for key, rec in items_second:
        partner = first_index.get(key)
        if partner is None:
            continue
        count += 1
        if len(examples) < EXAMPLE_LIMIT:
            examples.append(example_pair(partner, rec, text_fields, 1.0))
        if len(pairs) < PAIR_LIMIT:
            pairs.append([partner.id, rec.id, 1.0])

    denominator = len(items_second)
    rate = _rate(count, denominator)
    reverse_count = sum(1 for key, _ in items_first if key in second_keys)
    severity: Severity
    if count == 0:
        severity = "info"
    elif rate >= HIGH_RATE:
        severity = "high"
    else:
        severity = "medium"
    return Finding(
        check="exact",
        severity=severity,
        splits=[first, second],
        count=count,
        denominator=denominator,
        rate=rate,
        examples=examples,
        details={
            "scope": "cross",
            "pairs": pairs,
            "pairs_truncated": count > len(pairs),
            "measured_split": second,
            "reference_split": first,
            "reverse_count": reverse_count,
            "reverse_denominator": len(items_first),
            "reverse_rate": _rate(reverse_count, len(items_first)),
            "mask_numbers": mask_numbers,
            "key_method": KEY_METHOD,
        },
    )
