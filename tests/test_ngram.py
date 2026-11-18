"""Tests for benchlint.checks.ngram: 13-gram decontamination against a reference corpus."""

from __future__ import annotations

import random
import string

import pytest

from benchlint.checks.near import _tokens
from benchlint.checks.ngram import find_ngram_overlap
from benchlint.schema import NgramConfig, Record

CFG = NgramConfig()  # n = 13


def _words(n: int, seed: int) -> list[str]:
    """Return ``n`` random lowercase words, reproducible for ``seed``."""
    rng = random.Random(seed)
    return ["".join(rng.choices(string.ascii_lowercase, k=rng.randint(3, 8))) for _ in range(n)]


def _rec(rid: str, text: str, split: str = "test") -> Record:
    """Build a one-field eval record."""
    return Record(id=rid, split=split, fields={"text": text}, label=None)


def _grams(text: str, n: int) -> set[tuple[str, ...]]:
    """Return the distinct word n-grams of ``text`` as raw tuples (no hashing)."""
    tokens = _tokens(text)
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def _expected_fraction(item: str, reference: list[str], n: int) -> float:
    """Independent reference value: distinct item n-grams found in the reference, over all."""
    ref: set[tuple[str, ...]] = set()
    for doc in reference:
        ref |= _grams(doc, n)
    item_grams = _grams(item, n)
    return len(item_grams & ref) / len(item_grams)


def test_full_copy_is_flagged_with_fraction_one() -> None:
    """An eval item that repeats a reference passage has every n-gram in the reference."""
    passage = " ".join(_words(30, seed=1))
    finding = find_ngram_overlap([_rec("e0", passage)], [passage], ["text"], CFG)
    assert finding.check == "ngram"
    assert finding.count == 1
    assert finding.denominator == 1
    assert finding.rate == pytest.approx(1.0)
    assert finding.severity == "medium"
    assert finding.examples[0].similarity == pytest.approx(1.0)
    assert finding.examples[0].b_id == "reference:0"
    assert finding.examples[0].b_split == "reference"


def test_partial_overlap_matches_independent_fraction() -> None:
    """Half-copied items get the fraction of their distinct 13-grams found in the reference."""
    shared = _words(40, seed=2)
    reference = [" ".join(shared)]
    item = " ".join(shared[:20] + _words(20, seed=3))
    finding = find_ngram_overlap([_rec("e0", item)], reference, ["text"], CFG)
    expected = _expected_fraction(item, reference, CFG.n)
    assert 0.0 < expected < 1.0
    assert finding.count == 1
    assert finding.examples[0].similarity == pytest.approx(expected, abs=1e-12)


def test_disjoint_items_are_not_flagged() -> None:
    """Unrelated items give count 0, and the threshold counts are all 0."""
    reference = [" ".join(_words(200, seed=10))]
    items = [_rec(f"e{i}", " ".join(_words(50, seed=100 + i))) for i in range(20)]
    finding = find_ngram_overlap(items, reference, ["text"], CFG)
    assert finding.count == 0
    assert finding.denominator == 20
    assert finding.rate == 0.0
    assert finding.severity == "info"
    assert finding.examples == []
    assert finding.details["counts_at_fraction"] == {
        "any": 0,
        "ge_0.5": 0,
        "ge_0.8": 0,
        "ge_1.0": 0,
    }


def test_threshold_counts_are_monotone() -> None:
    """Counts at higher fraction thresholds never exceed counts at lower ones."""
    shared = _words(40, seed=4)
    reference = [" ".join(shared)]
    items = [
        _rec("full", " ".join(shared)),
        _rec("half", " ".join(shared[:20] + _words(20, seed=5))),
        _rec("none", " ".join(_words(40, seed=6))),
    ]
    counts = find_ngram_overlap(items, reference, ["text"], CFG).details["counts_at_fraction"]
    assert counts["any"] >= counts["ge_0.5"] >= counts["ge_0.8"] >= counts["ge_1.0"]
    assert counts["ge_1.0"] == 1
    assert counts["any"] == 2


def test_n_larger_than_text_is_not_evaluable() -> None:
    """Items with fewer than n tokens have no n-grams: not evaluable, rate 0.0, no error."""
    short = _rec("s", "only five words in here")
    finding = find_ngram_overlap([short], ["only five words in here"], ["text"], CFG)
    assert finding.denominator == 0
    assert finding.count == 0
    assert finding.rate == 0.0
    assert finding.details["n_items_too_short"] == 1
    assert finding.details["n_reference_ngrams"] == 0


def test_reference_shorter_than_n_gives_no_index() -> None:
    """A reference text shorter than n contributes no n-grams, so nothing is flagged."""
    item = " ".join(_words(30, seed=7))
    finding = find_ngram_overlap([_rec("e", item)], ["too short"], ["text"], CFG)
    assert finding.count == 0
    assert finding.details["n_reference_texts"] == 1
    assert finding.details["n_reference_ngrams"] == 0


def test_empty_reference_and_empty_items() -> None:
    """No reference texts and no eval items both give a clean zero finding."""
    item = " ".join(_words(30, seed=8))
    assert find_ngram_overlap([_rec("e", item)], [], ["text"], CFG).count == 0
    empty = find_ngram_overlap([], ["anything at all in this reference text here"], ["text"], CFG)
    assert empty.count == 0
    assert empty.denominator == 0
    assert empty.splits == []


def test_examples_are_capped_and_sorted() -> None:
    """At most five examples, highest fraction first, ties broken by item id."""
    passage = " ".join(_words(30, seed=9))
    items = [_rec(f"e{i:02d}", passage) for i in range(8)]
    finding = find_ngram_overlap(items, [passage], ["text"], CFG)
    assert finding.count == 8
    assert len(finding.examples) == 5
    assert [e.a_id for e in finding.examples] == ["e00", "e01", "e02", "e03", "e04"]


def test_reference_can_be_a_one_pass_iterable() -> None:
    """The reference is streamed: a generator works and is consumed once."""
    passage = " ".join(_words(30, seed=11))
    finding = find_ngram_overlap([_rec("e", passage)], (t for t in [passage]), ["text"], CFG)
    assert finding.count == 1


def test_invalid_n_raises() -> None:
    """n below 1 is rejected."""
    with pytest.raises(ValueError, match="at least 1"):
        find_ngram_overlap([], [], ["text"], NgramConfig(n=0))


def test_same_input_gives_identical_output() -> None:
    """Determinism: the hashed index gives the same finding on repeated runs."""
    shared = _words(40, seed=12)
    items = [_rec("a", " ".join(shared[:25] + _words(15, seed=13)))]

    def run() -> dict[str, object]:
        return find_ngram_overlap(items, [" ".join(shared)], ["text"], CFG).model_dump()

    assert run() == run()
