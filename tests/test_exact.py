"""Exact duplicate tests: within-split and cross-split, severity, normalisation. No network."""

from __future__ import annotations

from benchlint.checks.exact import find_exact_duplicates
from benchlint.schema import Finding, Record


def rec(rid: str, split: str, text: str, *, label: str | None = None) -> Record:
    return Record(id=rid, split=split, fields={"text": text}, label=label)


def by_scope(findings: list[Finding], scope: str, splits: list[str]) -> Finding:
    matches = [f for f in findings if f.details.get("scope") == scope and f.splits == splits]
    assert len(matches) == 1, f"expected one {scope} finding for {splits}, got {len(matches)}"
    return matches[0]


def test_empty_input_gives_no_findings() -> None:
    assert find_exact_duplicates([], ["text"]) == []


def test_single_clean_split_gives_info_with_zero_count() -> None:
    records = [rec("a", "test", "one"), rec("b", "test", "two")]
    findings = find_exact_duplicates(records, ["text"])
    assert len(findings) == 1
    f = findings[0]
    assert f.check == "exact"
    assert f.severity == "info"
    assert f.count == 0
    assert f.denominator == 2
    assert f.rate == 0.0
    assert f.examples == []


def test_within_split_duplicates_count_redundant_copies() -> None:
    records = [
        rec("a", "test", "same text"),
        rec("b", "test", "same text"),
        rec("c", "test", "same text"),
        rec("d", "test", "other"),
    ]
    findings = find_exact_duplicates(records, ["text"])
    f = by_scope(findings, "within", ["test"])
    assert f.count == 2  # three copies: two redundant
    assert f.denominator == 4
    assert f.rate == 0.5
    assert f.severity == "low"
    assert f.details["n_groups"] == 1
    assert f.details["largest_group"] == 3
    assert [(p.a_id, p.b_id) for p in f.examples] == [("a", "b"), ("a", "c")]
    assert all(p.similarity == 1.0 for p in f.examples)
    assert all(p.a_split == "test" and p.b_split == "test" for p in f.examples)


def test_cross_split_duplicate_is_high_at_one_percent() -> None:
    train = [rec(f"tr{i}", "train", f"train item {i}") for i in range(100)]
    test = [rec("te0", "test", "train item 3")] + [
        rec(f"te{i}", "test", f"fresh {i}") for i in range(1, 100)
    ]
    findings = find_exact_duplicates(train + test, ["text"])
    f = by_scope(findings, "cross", ["train", "test"])
    assert f.count == 1
    assert f.denominator == 100  # measured split is test, the later one
    assert f.rate == 0.01
    assert f.severity == "high"
    assert f.examples[0].a_id == "tr3"
    assert f.examples[0].b_id == "te0"
    assert f.details["measured_split"] == "test"
    assert f.details["reverse_count"] == 1
    assert f.details["reverse_denominator"] == 100


def test_cross_split_duplicate_below_one_percent_is_medium() -> None:
    train = [rec(f"tr{i}", "train", f"train item {i}") for i in range(300)]
    test = [rec("te0", "test", "train item 7")] + [
        rec(f"te{i}", "test", f"fresh {i}") for i in range(1, 200)
    ]
    f = by_scope(find_exact_duplicates(train + test, ["text"]), "cross", ["train", "test"])
    assert f.count == 1
    assert f.denominator == 200
    assert f.severity == "medium"  # 1/200 is 0.5%


def test_cross_split_without_overlap_is_info() -> None:
    findings = find_exact_duplicates(
        [rec("a", "train", "alpha"), rec("b", "test", "beta")], ["text"]
    )
    f = by_scope(findings, "cross", ["train", "test"])
    assert f.count == 0
    assert f.severity == "info"
    assert f.examples == []


def test_one_finding_per_split_and_per_pair() -> None:
    records = [
        rec("a", "train", "x"),
        rec("b", "validation", "x"),
        rec("c", "test", "y"),
    ]
    findings = find_exact_duplicates(records, ["text"])
    scopes = sorted((f.details["scope"], tuple(f.splits)) for f in findings)
    assert scopes == sorted(
        [
            ("within", ("train",)),
            ("within", ("validation",)),
            ("within", ("test",)),
            ("cross", ("train", "validation")),
            ("cross", ("train", "test")),
            ("cross", ("validation", "test")),
        ]
    )


def test_normalisation_makes_punctuation_and_case_irrelevant() -> None:
    records = [
        rec("a", "train", "Hello, World!"),
        rec("b", "test", "hello   world"),
    ]
    f = by_scope(find_exact_duplicates(records, ["text"]), "cross", ["train", "test"])
    assert f.count == 1


def test_unicode_equivalence_nfkc_and_casefold() -> None:
    records = [
        rec("a", "train", "Straße café"),
        rec("b", "test", "STRASSE café"),
        rec("c", "test", "ＡＢＣ"),  # fullwidth
        rec("d", "train", "abc"),
    ]
    f = by_scope(find_exact_duplicates(records, ["text"]), "cross", ["train", "test"])
    assert f.count == 2
    assert {p.b_id for p in f.examples} == {"b", "c"}


def test_field_boundaries_are_kept_for_multi_field_keys() -> None:
    fields_a = {"q": "a b", "a": "c"}
    fields_b = {"q": "a", "a": "b c"}
    records = [
        Record(id="x", split="train", fields=fields_a, label=None),
        Record(id="y", split="test", fields=fields_b, label=None),
    ]
    f = by_scope(find_exact_duplicates(records, ["q", "a"]), "cross", ["train", "test"])
    assert f.count == 0  # would be 1 if the unit separator were normalised away


def test_records_that_normalise_to_empty_are_skipped() -> None:
    records = [
        rec("a", "test", ""),
        rec("b", "test", "!!!"),
        rec("c", "test", "  "),
        rec("d", "test", "real text"),
    ]
    f = by_scope(find_exact_duplicates(records, ["text"]), "within", ["test"])
    assert f.details["skipped_empty"] == 3
    assert f.denominator == 1
    assert f.count == 0


def test_mask_numbers_merges_numeric_variants_only_when_enabled() -> None:
    records = [rec("a", "train", "Room 42"), rec("b", "test", "Room 7")]
    off = by_scope(find_exact_duplicates(records, ["text"]), "cross", ["train", "test"])
    assert off.count == 0
    on = by_scope(
        find_exact_duplicates(records, ["text"], mask_numbers=True),
        "cross",
        ["train", "test"],
    )
    assert on.count == 1
    assert on.details["mask_numbers"] is True


def test_examples_capped_at_five_and_text_truncated_to_200() -> None:
    long_text = "word " * 100  # far more than 200 characters
    train = [rec(f"tr{i}", "train", f"{long_text} {i}") for i in range(10)]
    test = [rec(f"te{i}", "test", f"{long_text} {i}") for i in range(10)]
    f = by_scope(find_exact_duplicates(train + test, ["text"]), "cross", ["train", "test"])
    assert f.count == 10
    assert len(f.examples) == 5
    for pair in f.examples:
        assert len(pair.a_text) <= 200
        assert len(pair.b_text) <= 200


def test_multi_field_text_is_joined_for_examples() -> None:
    records = [
        Record(id="a", split="train", fields={"q": "Q?", "a": "A."}, label=None),
        Record(id="b", split="test", fields={"q": "q?", "a": "a"}, label=None),
    ]
    f = by_scope(find_exact_duplicates(records, ["q", "a"]), "cross", ["train", "test"])
    assert f.count == 1
    assert f.examples[0].a_text == "Q? | A."
    assert f.examples[0].b_text == "q? | a"
