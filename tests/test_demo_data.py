"""Tests for demo_data: determinism, manifest correctness and planted-vs-found comparison.

The manifest is checked against the written files with independent code (plain JSON parsing,
re.findall tokens and set-based shingles), not with the generator's own helpers.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import pytest

from benchlint import demo_data
from benchlint.cli import load_audit_spec
from benchlint.demo_data import DemoManifest, build_demo, build_demo_dataset, compare_planted
from benchlint.demo_data.generate import LABEL_TOKENS
from benchlint.io import load_dataset
from benchlint.schema import (
    AuditReport,
    DatasetSpec,
    ExamplePair,
    Finding,
    ShortcutResult,
)

NEAR_THRESHOLD = 0.8
SHINGLE = 5


def _rows(path: Path) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        out[row["id"]] = row
    return out


def _tokens(text: str) -> list[str]:
    return re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold())


def _shingles(row: dict[str, str]) -> frozenset[tuple[str, ...]]:
    toks = _tokens(row["headline"] + " " + row["body"])
    return frozenset(tuple(toks[i : i + SHINGLE]) for i in range(len(toks) - SHINGLE + 1))


def _jaccard(a: frozenset[tuple[str, ...]], b: frozenset[tuple[str, ...]]) -> float:
    return len(a & b) / len(a | b)


@pytest.fixture(scope="module")
def bundle_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("demo")
    build_demo(out, seed=0)
    return out


@pytest.fixture(scope="module")
def manifest(bundle_dir: Path) -> DemoManifest:
    return DemoManifest.model_validate_json((bundle_dir / "manifest.json").read_text("utf-8"))


def _leak(manifest: DemoManifest, category: str) -> list[tuple[str, str]]:
    matches = [leak for leak in manifest.leaks if leak.category == category]
    assert len(matches) == 1, category
    return matches[0].pairs


# --- determinism -----------------------------------------------------------------------------


def test_same_seed_gives_identical_bytes(tmp_path: Path) -> None:
    a = tmp_path / "a"
    b = tmp_path / "b"
    build_demo(a, seed=0)
    build_demo(b, seed=0)
    names = sorted(str(p.relative_to(a)) for p in a.rglob("*") if p.is_file())
    assert names == sorted(str(p.relative_to(b)) for p in b.rglob("*") if p.is_file())
    for name in names:
        assert (a / name).read_bytes() == (b / name).read_bytes(), name


def test_different_seed_changes_the_data(tmp_path: Path) -> None:
    build_demo(tmp_path / "s0", seed=0)
    build_demo(tmp_path / "s1", seed=1)
    assert (tmp_path / "s0" / "data" / "train.jsonl").read_bytes() != (
        tmp_path / "s1" / "data" / "train.jsonl"
    ).read_bytes()


def test_manifest_files_match_checksums_and_row_counts(
    bundle_dir: Path, manifest: DemoManifest
) -> None:
    assert manifest.synthetic is True
    assert manifest.name.startswith("SYNTHETIC")
    counts = {f.path: f for f in manifest.files}
    for rel, expected_rows in [
        ("data/train.jsonl", 1000),
        ("data/test.jsonl", 300),
        ("reference/reference.jsonl", 150),
    ]:
        entry = counts[rel]
        data = (bundle_dir / rel).read_bytes()
        assert entry.sha256 == hashlib.sha256(data).hexdigest()
        assert entry.rows == expected_rows
        assert len(data.decode("utf-8").splitlines()) == expected_rows


# --- spec and loading ------------------------------------------------------------------------


def test_audit_yaml_loads_and_points_at_written_files(bundle_dir: Path) -> None:
    spec = load_audit_spec(bundle_dir / "audit.yaml")
    assert spec.dataset.name.startswith("SYNTHETIC")
    assert spec.dataset.source.kind == "jsonl"
    assert spec.checks.ngram.reference[0].id_or_path == "reference"
    assert [v.name for v in spec.checks.shortcut.views] == ["headline_only", "body_length"]


def test_build_demo_dataset_loads_through_io(tmp_path: Path) -> None:
    ds: DatasetSpec = build_demo_dataset(tmp_path, seed=0)
    splits = load_dataset(ds, base_dir=tmp_path)
    assert len(splits["train"]) == 1000
    assert len(splits["test"]) == 300
    assert all(rec.label in {"A", "B", "C", "D"} for rec in splits["test"])


# --- planted pairs, verified independently -------------------------------------------------


def test_exact_cross_pairs_are_identical_copies(bundle_dir: Path, manifest: DemoManifest) -> None:
    train = _rows(bundle_dir / "data" / "train.jsonl")
    test = _rows(bundle_dir / "data" / "test.jsonl")
    pairs = _leak(manifest, "exact_cross_split")
    assert len(pairs) == 12
    for train_id, test_id in pairs:
        a, b = train[train_id], test[test_id]
        assert (a["headline"], a["body"], a["label"]) == (b["headline"], b["body"], b["label"])


def test_within_train_pairs_and_label_conflicts(bundle_dir: Path, manifest: DemoManifest) -> None:
    train = _rows(bundle_dir / "data" / "train.jsonl")
    within = _leak(manifest, "exact_within_train")
    conflicts = set(map(tuple, _leak(manifest, "label_conflict")))
    assert len(within) == 8
    assert len(conflicts) == 6
    for a_id, b_id in within:
        a, b = train[a_id], train[b_id]
        assert (a["headline"], a["body"]) == (b["headline"], b["body"])
        if (a_id, b_id) in conflicts:
            assert a["label"] != b["label"]
        else:
            assert a["label"] == b["label"]


def test_near_pairs_are_above_threshold_and_not_exact(
    bundle_dir: Path, manifest: DemoManifest
) -> None:
    train = _rows(bundle_dir / "data" / "train.jsonl")
    test = _rows(bundle_dir / "data" / "test.jsonl")
    seen: dict[str, int] = defaultdict(int)
    for size in (1, 3, 5):
        pairs = _leak(manifest, f"near_edit_{size}word")
        assert len(pairs) == 9
        seen[str(size)] = len(pairs)
        for train_id, test_id in pairs:
            j = _jaccard(_shingles(train[train_id]), _shingles(test[test_id]))
            assert NEAR_THRESHOLD <= j < 1.0, (size, train_id, test_id, j)
    assert dict(seen) == {"1": 9, "3": 9, "5": 9}


def test_cross_split_pairs_above_threshold_are_exactly_the_planted_ones(
    bundle_dir: Path, manifest: DemoManifest
) -> None:
    """Every cross-split pair with Jaccard >= threshold is planted. Found with an inverted index
    so that the scan over 300 x 1000 items stays fast.
    """
    train = _rows(bundle_dir / "data" / "train.jsonl")
    test = _rows(bundle_dir / "data" / "test.jsonl")
    index: dict[tuple[str, ...], list[str]] = defaultdict(list)
    train_sh = {tid: _shingles(row) for tid, row in train.items()}
    for tid, sh in train_sh.items():
        for g in sh:
            index[g].append(tid)
    found: set[tuple[str, str]] = set()
    for sid, row in test.items():
        sh = _shingles(row)
        candidates = {tid for g in sh for tid in index.get(g, [])}
        for tid in candidates:
            if _jaccard(train_sh[tid], sh) >= NEAR_THRESHOLD:
                found.add((tid, sid))
    planted = set(map(tuple, _leak(manifest, "exact_cross_split")))
    for size in (1, 3, 5):
        planted |= set(map(tuple, _leak(manifest, f"near_edit_{size}word")))
    assert found == planted


def test_within_train_near_pairs_are_exactly_the_planted_exact_copies(
    bundle_dir: Path, manifest: DemoManifest
) -> None:
    train = _rows(bundle_dir / "data" / "train.jsonl")
    sh = {tid: _shingles(row) for tid, row in train.items()}
    index: dict[tuple[str, ...], list[str]] = defaultdict(list)
    for tid, grams in sh.items():
        for g in grams:
            index[g].append(tid)
    found: set[frozenset[str]] = set()
    for a, grams in sh.items():
        for b in {tid for g in grams for tid in index[g]}:
            if a < b and _jaccard(grams, sh[b]) >= NEAR_THRESHOLD:
                found.add(frozenset({a, b}))
    planted = {frozenset(p) for p in _leak(manifest, "exact_within_train")}
    assert found == planted


def test_ngram_planted_items_are_exactly_those_sharing_a_13_gram(
    bundle_dir: Path, manifest: DemoManifest
) -> None:
    test = _rows(bundle_dir / "data" / "test.jsonl")
    ref = _rows(bundle_dir / "reference" / "reference.jsonl")

    def grams(text: str) -> set[tuple[str, ...]]:
        toks = _tokens(text)
        return {tuple(toks[i : i + 13]) for i in range(len(toks) - 12)}

    ref_grams: set[tuple[str, ...]] = set()
    for row in ref.values():
        ref_grams |= grams(row["text"])
    shared = set()
    for tid, row in test.items():
        if grams(row["headline"] + " " + row["body"]) & ref_grams:
            shared.add(tid)
    planted_pairs = _leak(manifest, "ngram_reference")
    assert shared == {t for t, _ in planted_pairs}
    for test_id, ref_id in planted_pairs:
        assert grams(test[test_id]["headline"] + " " + test[test_id]["body"]) & grams(
            ref[ref_id]["text"]
        )


def test_shortcut_token_agreement_matches_manifest(
    bundle_dir: Path, manifest: DemoManifest
) -> None:
    test = _rows(bundle_dir / "data" / "test.jsonl")
    agree = 0
    for row in test.values():
        token_label = [lab for lab, tok in LABEL_TOKENS.items() if tok in row["headline"].split()]
        assert len(token_label) == 1
        agree += token_label[0] == row["label"]
    leak = next(x for x in manifest.leaks if x.category == "shortcut_token")
    assert leak.details["observed_agree_test"] == agree
    assert leak.details["observed_rate_test"] == pytest.approx(agree / 300)
    assert 0.85 <= agree / 300 <= 0.95


def test_manifest_counts_match_pairs(manifest: DemoManifest) -> None:
    for leak in manifest.leaks:
        if leak.pairs:
            assert leak.planted == len(leak.pairs), leak.category


# --- planted vs found ------------------------------------------------------------------------


def _report_with(findings: list[Finding], shortcuts: list[ShortcutResult]) -> AuditReport:
    spec = DatasetSpec(
        name="SYNTHETIC-demo",
        source={
            "kind": "jsonl",
            "id_or_path": "data",
            "config": None,
            "revision": None,
            "split_map": {"train": "train", "test": "test"},
        },
        text_fields=["headline", "body"],
        label_field="label",
        id_field="id",
        max_rows_per_split=None,
        seed=0,
    )
    return AuditReport(
        dataset="SYNTHETIC-demo",
        spec=spec,
        findings=findings,
        shortcuts=shortcuts,
        env_info={},
        runtime_s=0.1,
    )


def _f(check: str, splits: list[str], count: int, details: dict | None = None) -> Finding:
    return Finding(
        check=check,
        severity="high",
        splits=splits,
        count=count,
        denominator=300,
        rate=count / 300,
        examples=[
            ExamplePair(
                a_id="x",
                b_id="y",
                a_split=splits[0],
                b_split=splits[-1],
                similarity=1.0,
                a_text="",
                b_text="",
            )
        ],
        details=details or {},
    )


def test_compare_planted_uses_pairs_when_listed(manifest: DemoManifest) -> None:
    exact_pairs = _leak(manifest, "exact_cross_split")
    findings = [
        _f(
            "exact_across",
            ["train", "test"],
            len(exact_pairs),
            {"pairs": [list(p) for p in exact_pairs]},
        )
    ]
    rows = {r.category: r for r in compare_planted(manifest, _report_with(findings, []))}
    row = rows["exact_cross_split"]
    assert row.basis == "pairs"
    assert row.found == 12
    assert row.status == "found"


def test_compare_planted_partial_with_partial_pair_list(manifest: DemoManifest) -> None:
    exact_pairs = _leak(manifest, "exact_cross_split")
    listed = [list(p) for p in exact_pairs[:5]]
    findings = [_f("exact_across", ["train", "test"], 5, {"pairs": listed})]
    rows = {r.category: r for r in compare_planted(manifest, _report_with(findings, []))}
    assert rows["exact_cross_split"].status == "partial"
    assert rows["exact_cross_split"].found == 5


def test_compare_planted_near_is_not_attributed_without_pairs(manifest: DemoManifest) -> None:
    findings = [
        _f("near_cross", ["train", "test"], 38, {"scope": "cross_split", "n_near_pairs": 38})
    ]
    rows = compare_planted(manifest, _report_with(findings, []))
    per_bucket = [r for r in rows if r.category.startswith("near_edit")]
    assert all(r.status == "not attributed" and r.found is None for r in per_bucket)
    total = next(r for r in rows if r.category == "near_cross_split_total")
    assert total.planted == 39
    assert total.found == 38
    assert total.status == "partial"


def test_compare_planted_shortcut_flag(manifest: DemoManifest) -> None:
    shortcut = ShortcutResult(
        view="headline_only",
        metric="accuracy",
        value=0.9,
        ci_low=0.85,
        ci_high=0.95,
        majority=0.25,
        chance=0.25,
        n_train=1000,
        n_test=300,
        p_value=0.001,
        flagged=True,
    )
    rows = {r.category: r for r in compare_planted(manifest, _report_with([], [shortcut]))}
    assert rows["shortcut_token"].status == "found"
    unflagged = ShortcutResult(**{**shortcut.model_dump(), "flagged": False})
    rows = {r.category: r for r in compare_planted(manifest, _report_with([], [unflagged]))}
    assert rows["shortcut_token"].status == "missed"


def test_compare_planted_missing_findings_are_missed(manifest: DemoManifest) -> None:
    rows = {r.category: r for r in compare_planted(manifest, _report_with([], []))}
    assert rows["exact_cross_split"].status == "missed"
    assert rows["ngram_reference"].status == "missed"
    assert rows["near_cross_split_total"].status == "missed"


def test_public_exports() -> None:
    assert demo_data.SYNTHETIC_NAME == "SYNTHETIC-demo"


def test_label_conflicts_with_three_of_six_pairs_listed_is_partial(manifest: DemoManifest) -> None:
    """Pair-level recall: three of the six planted label pairs gives 3 found, status partial."""
    pairs = _leak(manifest, "label_conflict")
    assert len(pairs) == 6
    listed = [[a, b, 1.0] for a, b in pairs[:3]]
    findings = [_f("labels", ["train"], 6, {"pairs": listed})]
    rows = {r.category: r for r in compare_planted(manifest, _report_with(findings, []))}
    row = rows["label_conflict"]
    assert row.basis == "pairs"
    assert row.planted == 6
    assert row.found == 3
    assert row.status == "partial"


def test_label_conflict_without_pair_listing_is_not_attributed(manifest: DemoManifest) -> None:
    findings = [_f("labels", ["train"], 12, {})]
    rows = {r.category: r for r in compare_planted(manifest, _report_with(findings, []))}
    assert rows["label_conflict"].status == "not attributed"
    assert rows["label_conflict"].found is None
