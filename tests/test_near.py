"""Tests for benchlint.checks.near: planted near-duplicates, false positives, determinism."""

from __future__ import annotations

import random
import string

import pytest

from benchlint.checks.near import _tokens, find_near_duplicates, shingle_set
from benchlint.schema import Finding, NearConfig, Record

CFG = NearConfig()  # threshold 0.8, num_perm 128, 5-word shingles


def _words(n: int, seed: int) -> list[str]:
    """Return ``n`` random lowercase words, reproducible for ``seed``."""
    rng = random.Random(seed)
    return ["".join(rng.choices(string.ascii_lowercase, k=rng.randint(3, 8))) for _ in range(n)]


def _rec(rid: str, split: str, text: str) -> Record:
    """Build a one-field record."""
    return Record(id=rid, split=split, fields={"text": text}, label=None)


def _edit(words: list[str], k: int, mode: str, seed: int) -> list[str]:
    """Apply ``k`` word edits at distinct positions, each using a token absent from the base.

    ``sub`` replaces, ``del`` removes and ``ins`` inserts. Every edit touches a distinct
    original token or gap, so the word-level edit distance is k for random-word text.
    """
    rng = random.Random(seed)
    out = list(words)
    if mode == "sub":
        for j, pos in enumerate(rng.sample(range(len(words)), k)):
            out[pos] = f"zq{j}x"
        return out
    if mode == "del":
        for pos in sorted(rng.sample(range(len(words)), k), reverse=True):
            del out[pos]
        return out
    if mode == "ins":
        for j, pos in enumerate(sorted(rng.sample(range(len(words) + 1), k), reverse=True)):
            out.insert(pos, f"zq{j}x")
        return out
    raise ValueError(mode)


def _jaccard(a: str, b: str) -> float:
    """Exact Jaccard of the 5-word shingle sets of two texts (independent of the module)."""
    sa = shingle_set(_tokens(a), 5)
    sb = shingle_set(_tokens(b), 5)
    return len(sa & sb) / len(sa | sb)


def _cross(findings: list[Finding]) -> Finding:
    """Return the cross-split finding."""
    return next(f for f in findings if f.details["scope"] == "cross_split")


@pytest.mark.parametrize("length", [100, 300])
@pytest.mark.parametrize("mode", ["sub", "del", "ins"])
@pytest.mark.parametrize("k", [1, 3, 5, 10])
def test_planted_word_edits_follow_exact_jaccard(length: int, mode: str, k: int) -> None:
    """A planted edited copy is reported only when its exact Jaccard reaches the threshold.

    Verification is exact, so a pair below the threshold is never reported. Recall is
    probabilistic, so the test only requires detection for pairs at Jaccard 0.85 or above.
    Observed behaviour: detection depends on length. With 100 words, 3 or more edits drop
    Jaccard below 0.8; with 300 words, up to about 5 edits still clear it.
    """
    base = " ".join(_words(length, seed=length + k))
    edited = " ".join(_edit(base.split(), k, mode, seed=1000 + k))
    jac = _jaccard(base, edited)
    findings = find_near_duplicates(
        [_rec("a", "train", base), _rec("b", "test", edited)], ["text"], CFG, seed=0
    )
    finding = _cross(findings)
    if jac < CFG.threshold:
        assert finding.count == 0
    if jac >= 0.85:
        assert finding.count == 2
    if finding.count:
        assert finding.examples[0].similarity == pytest.approx(jac, abs=1e-12)


def test_identical_texts_across_splits_are_always_found() -> None:
    """Identical texts have identical signatures, so LSH always returns them."""
    text = " ".join(_words(60, seed=7))
    records = [_rec(f"tr{i}", "train", text) for i in range(6)]
    records += [_rec(f"te{i}", "test", text) for i in range(6)]
    cross = _cross(find_near_duplicates(records, ["text"], CFG))
    assert cross.count == 12
    assert cross.denominator == 12
    assert cross.rate == pytest.approx(1.0)
    assert cross.details["near_pairs_by_split_pair"] == {"test|train": 36}


def test_examples_are_capped_and_sorted_by_similarity() -> None:
    """At most five examples, highest similarity first, with split labels set."""
    base = " ".join(_words(60, seed=11))
    edited = " ".join(_edit(base.split(), 1, "sub", seed=3))
    records = [_rec(f"tr{i}", "train", base) for i in range(4)]
    records += [_rec(f"te{i}", "test", base) for i in range(4)]
    records += [_rec("te_edit", "test", edited)]
    cross = _cross(find_near_duplicates(records, ["text"], CFG))
    assert 1 <= len(cross.examples) <= 5
    sims = [e.similarity for e in cross.examples]
    assert sims == sorted(sims, reverse=True)
    assert all(e.a_split != e.b_split for e in cross.examples)
    assert all(len(e.a_text) <= 200 for e in cross.examples)


def test_unrelated_texts_produce_no_near_duplicates() -> None:
    """False-positive check: 200 unrelated random texts give zero flags in both scopes."""
    records = [
        _rec(f"r{i}", "train" if i % 2 else "test", " ".join(_words(60, seed=5000 + i)))
        for i in range(200)
    ]
    findings = find_near_duplicates(records, ["text"], CFG, seed=0)
    assert [f.details["scope"] for f in findings] == ["cross_split", "within_split"]
    for finding in findings:
        assert finding.check == "near"
        assert finding.count == 0
        assert finding.rate == 0.0
        assert finding.severity == "info"
        assert finding.examples == []
        assert finding.details["n_near_pairs"] == 0
        # Every LSH candidate that did appear is below the threshold.
        assert (
            finding.details["n_candidate_pairs_below_threshold"]
            == (finding.details["n_candidate_pairs"])
        )


def test_template_phrasing_is_flagged_and_shows_in_histogram() -> None:
    """Template-driven texts share most shingles, so they are flagged. The histogram exposes it.

    This is the false-positive risk named in DECISION.md section 2.3. Only the template's
    shared shingles are checked here, not whether the flags are wrong.
    """
    template = " ".join(_words(30, seed=21))
    records = [_rec(f"t{i}", "train", f"{template} unique{i}x") for i in range(10)]
    records += [_rec(f"u{i}", "test", f"{template} other{i}y") for i in range(10)]
    cross = _cross(find_near_duplicates(records, ["text"], CFG))
    assert cross.count > 0
    counts = cross.details["jaccard_histogram"]["counts"]
    assert sum(counts[8:]) > 0  # some candidate pairs sit at Jaccard 0.8 or above
    assert all(e.similarity >= CFG.threshold for e in cross.examples)


def test_normalisation_ignores_case_and_punctuation() -> None:
    """Texts that differ only in case and punctuation normalise to the same shingles."""
    assert _tokens("Hello,  WORLD!  Don't") == ["hello", "world", "dont"]
    a = "The quick brown fox jumps over the lazy dog today."
    b = "THE QUICK, BROWN FOX! jumps over the lazy dog today"
    records = [_rec("a", "train", a), _rec("b", "test", b)]
    assert _cross(find_near_duplicates(records, ["text"], CFG)).count == 2


def test_short_texts_use_one_shingle() -> None:
    """Texts shorter than the shingle size compare as one shingle: equal means Jaccard 1."""
    records = [_rec("a", "train", "two words"), _rec("b", "test", "Two  words!")]
    cross = _cross(find_near_duplicates(records, ["text"], CFG))
    assert cross.count == 2
    assert cross.examples[0].similarity == pytest.approx(1.0)


def test_same_seed_gives_identical_output() -> None:
    """Determinism: the same records and seed give identical findings."""
    base = " ".join(_words(80, seed=31))
    edited = " ".join(_edit(base.split(), 1, "sub", seed=4))
    records = [_rec("a", "train", base), _rec("b", "test", edited), _rec("c", "test", base)]
    first = [f.model_dump() for f in find_near_duplicates(records, ["text"], CFG, seed=42)]
    second = [f.model_dump() for f in find_near_duplicates(records, ["text"], CFG, seed=42)]
    assert first == second


def test_empty_input_gives_no_findings() -> None:
    """No records, no findings."""
    assert find_near_duplicates([], ["text"], CFG) == []


def test_empty_text_records_are_not_evaluable() -> None:
    """Records with empty text are excluded from the denominator and never flagged."""
    records = [_rec("a", "train", ""), _rec("b", "test", "   "), _rec("c", "test", "!!!")]
    findings = find_near_duplicates(records, ["text"], CFG)
    for finding in findings:
        assert finding.count == 0
        assert finding.denominator == 0
        assert finding.rate == 0.0


def test_single_split_gives_only_within_split_finding() -> None:
    """A cross-split finding needs at least two splits; within-split is always reported."""
    base = " ".join(_words(60, seed=41))
    records = [_rec("a", "train", base), _rec("b", "train", base)]
    findings = find_near_duplicates(records, ["text"], CFG)
    assert [f.details["scope"] for f in findings] == ["within_split"]
    assert findings[0].count == 2
    assert findings[0].splits == ["train"]


def test_missing_text_field_raises() -> None:
    """A record without one of the text fields is an error, not a silent empty text."""
    records = [Record(id="a", split="train", fields={"other": "x"}, label=None)]
    with pytest.raises(ValueError, match="lacks fields"):
        find_near_duplicates(records, ["text"], CFG)
