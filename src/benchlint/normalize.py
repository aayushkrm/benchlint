"""Text normalisation shared by all checks.

Steps, in this order (pre-registered method; do not change silently):

1. NFKC Unicode normalisation.
2. casefold (not lower, so ``"ß"`` becomes ``"ss"``).
3. Punctuation strip: every character of Unicode category ``P*`` is deleted. Symbols
   (category ``S*``, for example ``$``, ``+`` or ``%``) are kept, so arithmetic text is
   not collapsed.
4. Optional number masking: every run of Unicode decimal digits becomes a single ``0``.
5. Whitespace collapse: runs of whitespace become one space; leading and trailing space
   is removed.

Fields are normalised one at a time (see :func:`normalize_fields`). The unit separator
``\\x1f`` counts as whitespace for ``str.split``, so normalising a concatenated string would
erase field boundaries and make ``("a b", "c")`` equal to ``("a", "b c")``.
"""

from __future__ import annotations

import functools
import re
import sys
import unicodedata
from collections.abc import Mapping, Sequence

FIELD_SEPARATOR = "\x1f"
"""Unit separator placed between fields in :func:`fields_to_text` and exact keys."""

_DIGIT_RUN = re.compile(r"\d+")


@functools.cache
def _punctuation_pattern() -> re.Pattern[str]:
    """Return a compiled character class of every Unicode ``P*`` code point (built once)."""
    chars = [
        chr(cp) for cp in range(sys.maxunicode + 1) if unicodedata.category(chr(cp)).startswith("P")
    ]
    return re.compile("[" + "".join(re.escape(c) for c in chars) + "]")


def normalize_text(text: str, *, mask_numbers: bool = False) -> str:
    """Return the normalised form of ``text``.

    The steps are described in the module docstring. If ``mask_numbers`` is true, every run
    of digits is replaced by a single ``0``. Empty or whitespace-only input returns ``""``.
    """
    out = unicodedata.normalize("NFKC", text).casefold()
    out = _punctuation_pattern().sub("", out)
    if mask_numbers:
        out = _DIGIT_RUN.sub("0", out)
    return " ".join(out.split())


def word_tokens(text: str, *, mask_numbers: bool = False) -> list[str]:
    """Word tokens for shingles and n-grams: punctuation becomes a space, not nothing.

    Steps 1, 2, 4 and 5 of :func:`normalize_text` are the same. Step 3 differs: each ``P*``
    character becomes a space, so "well-known" gives two words and "U.S." gives "u s". Symbols
    (``S*``) are kept, as in ``normalize_text``. This is the tokenisation used by the near
    duplicate, label and n-gram checks; the exact-duplicate key still uses ``normalize_text``.
    """
    out = unicodedata.normalize("NFKC", text).casefold()
    out = _punctuation_pattern().sub(" ", out)
    if mask_numbers:
        out = _DIGIT_RUN.sub("0", out)
    return out.split()


def normalize_fields(
    fields: Mapping[str, str], keys: Sequence[str], *, mask_numbers: bool = False
) -> list[str]:
    """Normalise each named field separately and return the results in the given order.

    Missing keys raise ``KeyError``; the caller decides what to do with them.
    """
    return [normalize_text(fields[k], mask_numbers=mask_numbers) for k in keys]


def fields_to_text(fields: Mapping[str, str], keys: Sequence[str]) -> str:
    """Concatenate the named fields (in the given order) with a unit-separator character.

    The result is not normalised. Missing keys raise ``KeyError``.
    """
    return FIELD_SEPARATOR.join(fields[k] for k in keys)
