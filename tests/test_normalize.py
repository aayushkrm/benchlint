"""Tests for the normalisation steps and the field helpers (no I/O, no network)."""

from __future__ import annotations

import pytest

from benchlint.normalize import (
    FIELD_SEPARATOR,
    fields_to_text,
    normalize_fields,
    normalize_text,
)


def test_empty_and_whitespace_only_give_empty_string() -> None:
    assert normalize_text("") == ""
    assert normalize_text("   \t\n  ") == ""


def test_nfkc_folds_fullwidth_and_ligatures() -> None:
    assert normalize_text("ＡＢＣ１２３") == "abc123"
    assert normalize_text("ﬁne") == "fine"


def test_casefold_uses_full_folding_for_sharp_s() -> None:
    assert normalize_text("Straße") == "strasse"
    assert normalize_text("HELLO World") == "hello world"


def test_composed_and_decomposed_accents_are_equal() -> None:
    composed = "café"  # e with acute, one code point
    decomposed = "café"  # e plus combining acute
    assert normalize_text(composed) == normalize_text(decomposed) == "café"


def test_punctuation_is_stripped() -> None:
    assert normalize_text("Hello, World!") == "hello world"
    assert normalize_text('(a) "quoted" — dash; colon:') == "a quoted dash colon"


def test_punctuation_strip_joins_split_tokens() -> None:
    # Deletion, not replacement: "e-mail" becomes "email" and "1,000" becomes "1000".
    assert normalize_text("e-mail") == "email"
    assert normalize_text("1,000") == "1000"


def test_symbols_are_kept() -> None:
    # Currency and arithmetic symbols are category S, not P, so they survive. "%" is Po
    # (other punctuation) and is stripped.
    assert normalize_text("$5 + 3 = 8%") == "$5 + 3 = 8"


def test_whitespace_collapses_including_nbsp_and_newlines() -> None:
    assert normalize_text("a\t\tb\n\nc d") == "a b c d"
    assert normalize_text("  leading and trailing  ") == "leading and trailing"


def test_unicode_text_is_preserved() -> None:
    assert normalize_text("日本語 テキスト") == "日本語 テキスト"
    assert normalize_text("Привет, мир!") == "привет мир"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Room 42 has 7 chairs", "room 0 has 0 chairs"),
        ("1,000 apples", "0 apples"),
        ("no digits here", "no digits here"),
        ("٣ arabic digit", "0 arabic digit"),
    ],
)
def test_mask_numbers_replaces_each_digit_run(text: str, expected: str) -> None:
    assert normalize_text(text, mask_numbers=True) == expected


def test_mask_numbers_off_keeps_digits() -> None:
    assert normalize_text("Room 42") == "room 42"


def test_fields_to_text_joins_in_given_order_with_separator() -> None:
    fields = {"a": "first", "b": "second", "c": "third"}
    assert fields_to_text(fields, ["c", "a"]) == f"third{FIELD_SEPARATOR}first"


def test_fields_to_text_missing_key_raises() -> None:
    with pytest.raises(KeyError):
        fields_to_text({"a": "x"}, ["a", "missing"])


def test_normalize_fields_normalises_each_field_separately() -> None:
    fields = {"q": "Hello,  World!", "a": "42"}
    assert normalize_fields(fields, ["q", "a"]) == ["hello world", "42"]
    assert normalize_fields(fields, ["a"], mask_numbers=True) == ["0"]


def test_separator_is_not_normalised_away_inside_a_field_join() -> None:
    # Regression guard: \x1f counts as whitespace for str.split. Normalising the joined text
    # would merge ("a b", "c") with ("a", "b c"). Per-field normalisation keeps them apart.
    joined_then_normalised = normalize_text(fields_to_text({"x": "a b", "y": "c"}, ["x", "y"]))
    other = normalize_text(fields_to_text({"x": "a", "y": "b c"}, ["x", "y"]))
    assert joined_then_normalised == other  # this is the collision the stub design would cause
    assert normalize_fields({"x": "a b", "y": "c"}, ["x", "y"]) != normalize_fields(
        {"x": "a", "y": "b c"}, ["x", "y"]
    )


def test_word_tokens_split_on_punctuation_and_keep_symbols() -> None:
    from benchlint.normalize import word_tokens

    assert word_tokens("well-known") == ["well", "known"]
    assert word_tokens("U.S. Don't") == ["u", "s", "don", "t"]
    # "$" and "+" are symbols (kept); "%" is punctuation (Po), so it becomes a space.
    assert word_tokens("cost $5 + 3%") == ["cost", "$5", "+", "3"]
    assert word_tokens("ab 12 cd", mask_numbers=True) == ["ab", "0", "cd"]
    assert word_tokens("") == []


def test_exact_key_normalisation_is_unchanged_by_word_tokens() -> None:
    from benchlint.normalize import normalize_text

    assert normalize_text("well-known") == "wellknown"
