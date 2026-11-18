"""Planted-leak manifest for the SYNTHETIC demo, and the planted-vs-found comparison.

The manifest is written next to the demo data as ``manifest.json``. ``compare_planted`` matches
it against an ``AuditReport`` so that ``benchlint demo`` can print what was planted next to
what the checks found.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from benchlint.schema import AuditReport, Finding

SYNTHETIC_NAME = "SYNTHETIC-demo"


class PlantedLeak(BaseModel):
    """One planted category: how many items or pairs were planted, and which pairs.

    ``pairs`` holds ``(a_id, b_id)`` tuples for pair-level categories. Shortcut categories carry
    their numbers in ``details`` instead.
    """

    category: str
    check: Literal["exact", "labels", "near", "ngram", "shortcut"]
    description: str
    planted: int
    splits: list[str]
    pairs: list[tuple[str, str]] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)


class DemoFile(BaseModel):
    """A data file written by the generator, with its row count and checksum."""

    path: str
    rows: int
    sha256: str


class DemoManifest(BaseModel):
    """Everything planted in the demo dataset. Contains no timestamps, so it is reproducible."""

    name: str = SYNTHETIC_NAME
    synthetic: Literal[True] = True
    seed: int
    files: list[DemoFile]
    leaks: list[PlantedLeak]


class LeakRow(BaseModel):
    """One line of the planted-vs-found table.

    ``found`` is None when the report cannot attribute a count to this category (for example
    a near-duplicate finding without pair details).
    """

    category: str
    check: str
    planted: int
    found: int | None
    status: Literal["found", "partial", "missed", "not attributed"]
    basis: Literal["pairs", "count", "flag"]
    note: str = ""


def _pairs_of(finding: Finding) -> set[frozenset[str]] | None:
    """Pairs listed in ``finding.details['pairs']``, or None if the check did not list them."""
    raw = finding.details.get("pairs")
    if not isinstance(raw, list):
        return None
    out: set[frozenset[str]] = set()
    for item in raw:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            # [a_id, b_id] or [a_id, b_id, similarity] as written by the near check.
            out.add(frozenset({str(item[0]), str(item[1])}))
        elif isinstance(item, dict) and "a_id" in item and "b_id" in item:
            out.add(frozenset({str(item["a_id"]), str(item["b_id"])}))
        else:
            return None
    return out


def _matching_findings(findings: list[Finding], leak: PlantedLeak) -> list[Finding]:
    """Findings that belong to a planted category, selected by check name and split pattern."""
    selected: list[Finding] = []
    cross = len(set(leak.splits)) > 1
    for f in findings:
        name = f.check.lower()
        if leak.check == "exact":
            if "exact" in name and (len(set(f.splits)) > 1) == cross:
                selected.append(f)
        elif leak.check == "labels":
            if "label" in name:
                selected.append(f)
        elif leak.check == "near":
            if "near" in name:
                selected.append(f)
        elif leak.check == "ngram":
            if "ngram" in name:
                selected.append(f)
    return selected


def _status(found: int, planted: int) -> Literal["found", "partial", "missed"]:
    """Status from a count: all planted found, some found, or none found."""
    if found >= planted:
        return "found"
    if found > 0:
        return "partial"
    return "missed"


def compare_planted(manifest: DemoManifest, report: AuditReport) -> list[LeakRow]:
    """Compare each planted category with what the report contains.

    Pair-level recall is used when the check lists its pairs in ``details['pairs']``. Otherwise
    the finding count is compared with the planted count. Near-duplicate categories without
    pair details are reported as "not attributed", never as found.
    """
    rows: list[LeakRow] = []
    for leak in manifest.leaks:
        if leak.check == "shortcut":
            view = str(leak.details.get("view", ""))
            flagged = any(s.view == view and s.flagged for s in report.shortcuts)
            rows.append(
                LeakRow(
                    category=leak.category,
                    check=leak.check,
                    planted=leak.planted,
                    found=1 if flagged else 0,
                    status="found" if flagged else "missed",
                    basis="flag",
                    note=f"view {view}: flagged" if flagged else f"view {view}: not flagged",
                )
            )
            continue

        matches = _matching_findings(report.findings, leak)
        total_count = sum(f.count for f in matches)
        planted_pairs = {frozenset(p) for p in leak.pairs}
        pair_sets = [_pairs_of(f) for f in matches]
        if not matches and leak.check in ("exact", "labels", "near"):
            # No finding of this kind at all: zero of the planted pairs were listed.
            rows.append(
                LeakRow(
                    category=leak.category,
                    check=leak.check,
                    planted=len(planted_pairs),
                    found=0,
                    status="missed",
                    basis="pairs",
                    note="no finding of this check kind; 0 planted pairs listed",
                )
            )
            continue
        if matches and all(ps is not None for ps in pair_sets):
            found_pairs: set[frozenset[str]] = set()
            for ps in pair_sets:
                if ps is not None:
                    found_pairs |= ps
            hit = len(planted_pairs & found_pairs)
            rows.append(
                LeakRow(
                    category=leak.category,
                    check=leak.check,
                    planted=len(planted_pairs),
                    found=hit,
                    status=_status(hit, len(planted_pairs)),
                    basis="pairs",
                    note=f"{hit} of {len(planted_pairs)} planted pairs listed by the check",
                )
            )
            continue

        if leak.check in ("near", "labels", "exact"):
            # These checks list their pairs; without a listing the row is not attributed,
            # never counted (items and pairs are different units).
            rows.append(
                LeakRow(
                    category=leak.category,
                    check=leak.check,
                    planted=len(planted_pairs),
                    found=None,
                    status="not attributed",
                    basis="count",
                    note="check lists no pairs, so the row cannot be attributed",
                )
            )
            continue

        # n-gram: the planted count is the number of quoted test items, and the check counts
        # test items with a match, so both sides are items.
        rows.append(
            LeakRow(
                category=leak.category,
                check=leak.check,
                planted=leak.planted,
                found=total_count,
                status=_status(total_count, leak.planted),
                basis="count",
                note=f"items: check count {total_count} of {leak.planted} quoted test items",
            )
        )

    near_leaks = [leak for leak in manifest.leaks if leak.check == "near"]
    if near_leaks:
        # Exact copies across splits are also near pairs, so they belong in the expected total.
        exact_cross = [leak for leak in manifest.leaks if leak.category == "exact_cross_split"]
        expected = sum(leak.planted for leak in near_leaks) + sum(
            leak.planted for leak in exact_cross
        )
        cross = [
            f
            for f in report.findings
            if "near" in f.check.lower() and f.details.get("scope") == "cross_split"
        ]
        if cross:
            found = sum(int(f.details.get("n_near_pairs", f.count)) for f in cross)
            status = _status(found, expected)
            note = (
                f"{found} cross-split near pairs reported; expected {expected} "
                "(planted edits plus exact copies)"
            )
        else:
            found = 0
            status = "missed"
            note = "no cross-split near finding"
        rows.append(
            LeakRow(
                category="near_cross_split_total",
                check="near",
                planted=expected,
                found=found,
                status=status,
                basis="count",
                note=note,
            )
        )
    return rows
