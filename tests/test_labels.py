"""Label conflict tests: exact duplicates, near mode, and missing labels. No network."""

from __future__ import annotations

from typing import Any

import pytest

from benchlint.checks.labels import find_label_conflicts
from benchlint.schema import Finding, NearConfig, Record


def rec(rid: str, split: str, text: str, label: Any) -> Record:
    return Record(id=rid, split=split, fields={"text": text}, label=label)


def _one(findings: list[Finding]) -> Finding:
    assert len(findings) == 1
    assert findings[0].check == "labels"
    return findings[0]


def test_no_labelled_records_gives_no_findings() -> None:
    assert find_label_conflicts([], ["text"]) == []
    unlabelled = [rec("a", "train", "x", None), rec("b", "test", "x", None)]
    assert find_label_conflicts(unlabelled, ["text"]) == []


def test_cross_split_exact_duplicate_with_different_labels_is_a_conflict() -> None:
    records = [
        rec("tr0", "train", "The cat sat.", "pos"),
        rec("tr1", "train", "Something else.", "neg"),
        rec("tr2", "train", "A third item.", "pos"),
        rec("te0", "test", "the cat sat", "neg"),  # same normalised text, other label
    ]
    f = _one(find_label_conflicts(records, ["text"]))
    assert f.count == 2  # both members of the conflicting group
    assert f.denominator == 4
    assert f.rate == 0.5
    assert f.severity == "high"
    assert f.splits == ["train", "test"]
    assert f.details["mode"] == "exact"
    assert f.details["n_conflict_groups"] == 1
    assert f.details["label_counts_in_conflicts"] == {"pos": 1, "neg": 1}
    assert [(p.a_id, p.b_id) for p in f.examples] == [("tr0", "te0")]
    assert f.examples[0].similarity == 1.0


def test_consistent_duplicates_are_not_conflicts() -> None:
    records = [
        rec("a", "train", "same", "pos"),
        rec("b", "test", "same", "pos"),
    ]
    f = _one(find_label_conflicts(records, ["text"]))
    assert f.count == 0
    assert f.severity == "info"
    assert f.details["n_groups_checked"] == 1
    assert f.details["n_conflict_groups"] == 0
    assert f.examples == []


def test_within_split_conflict_is_detected() -> None:
    records = [
        rec("a", "test", "dup", 0),
        rec("b", "test", "dup", 1),
        rec("c", "test", "unique", 0),
        rec("d", "test", "also unique", 1),
    ]
    f = _one(find_label_conflicts(records, ["text"]))
    assert f.count == 2
    assert f.splits == ["test"]
    assert f.severity == "high"  # 2 of 4 labelled records


def test_missing_label_does_not_count_as_conflict() -> None:
    records = [
        rec("a", "train", "same", "pos"),
        rec("b", "test", "same", None),
        rec("c", "test", "same", ""),  # empty label is treated as missing
    ]
    f = _one(find_label_conflicts(records, ["text"]))
    assert f.count == 0
    assert f.denominator == 1  # only record a is labelled
    assert f.severity == "info"


def test_label_comparison_is_by_text_so_int_and_str_agree() -> None:
    records = [rec("a", "train", "same", 1), rec("b", "test", "same", "1")]
    f = _one(find_label_conflicts(records, ["text"]))
    assert f.count == 0


def test_medium_severity_below_one_percent() -> None:
    records = [rec(f"x{i}", "train", f"item {i}", i % 2) for i in range(199)]
    records.append(rec("dup-a", "train", "shared text", "pos"))
    records.append(rec("dup-b", "test", "shared text", "neg"))
    f = _one(find_label_conflicts(records, ["text"]))
    assert f.count == 2
    assert f.denominator == 201
    assert f.severity == "medium"


def test_examples_are_capped_at_five() -> None:
    train = [rec(f"tr{i}", "train", f"text {i}", "pos") for i in range(8)]
    test = [rec(f"te{i}", "test", f"text {i}", "neg") for i in range(8)]
    f = _one(find_label_conflicts(train + test, ["text"]))
    assert f.details["n_conflict_groups"] == 8
    assert len(f.examples) == 5


# 100 words, one word edited: 96 shingles each, 91 shared, so Jaccard = 91/101 (about 0.90).
# A longer text keeps the pair well above the threshold used below, so the test checks the
# grouping logic and not MinHash recall near the boundary.
WORDS = [f"w{i}" for i in range(100)]
BASE = " ".join(WORDS)
ONE_WORD_CHANGED = " ".join(WORDS[:50] + ["changed"] + WORDS[51:])
UNRELATED = " ".join(f"u{i}" for i in range(100))


def test_exact_mode_does_not_group_near_duplicates() -> None:
    records = [rec("a", "train", BASE, "pos"), rec("b", "test", ONE_WORD_CHANGED, "neg")]
    f = _one(find_label_conflicts(records, ["text"]))
    assert f.count == 0


def test_near_mode_groups_one_word_edit_above_threshold() -> None:
    records = [rec("a", "train", BASE, "pos"), rec("b", "test", ONE_WORD_CHANGED, "neg")]
    near = NearConfig(threshold=0.7, num_perm=128, shingle=5)
    f = _one(find_label_conflicts(records, ["text"], near=near))
    assert f.details["mode"] == "near"
    assert f.count == 2
    assert f.severity == "high"
    assert f.examples[0].similarity == pytest.approx(91 / 101, abs=1e-9)


def test_near_mode_respects_threshold() -> None:
    records = [rec("a", "train", BASE, "pos"), rec("b", "test", ONE_WORD_CHANGED, "neg")]
    strict = NearConfig(threshold=0.95, num_perm=128, shingle=5)
    f = _one(find_label_conflicts(records, ["text"], near=strict))
    assert f.count == 0


def test_near_mode_keeps_unrelated_texts_apart() -> None:
    records = [rec("a", "train", BASE, "pos"), rec("b", "test", UNRELATED, "neg")]
    near = NearConfig(threshold=0.7, num_perm=128, shingle=5)
    f = _one(find_label_conflicts(records, ["text"], near=near))
    assert f.count == 0


def test_near_mode_is_deterministic_for_fixed_seed() -> None:
    records = [rec("a", "train", BASE, "pos"), rec("b", "test", ONE_WORD_CHANGED, "neg")]
    near = NearConfig(threshold=0.7, num_perm=128, shingle=5)
    first = _one(find_label_conflicts(records, ["text"], near=near, seed=7))
    again = _one(find_label_conflicts(records, ["text"], near=near, seed=7))
    assert first.model_dump() == again.model_dump()
    assert first.details["seed"] == 7


def test_mask_numbers_applies_before_grouping() -> None:
    records = [rec("a", "train", "Room 42", "pos"), rec("b", "test", "Room 7", "neg")]
    assert _one(find_label_conflicts(records, ["text"])).count == 0
    assert _one(find_label_conflicts(records, ["text"], mask_numbers=True)).count == 2
