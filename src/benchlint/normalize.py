"""Text normalisation shared by all checks.

Steps (in order): NFKC, casefold, punctuation strip, whitespace collapse, and optional
number masking. The exact steps are part of the pre-registered method and must not change
silently; see docs/design.md once written.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence


def normalize_text(text: str, *, mask_numbers: bool = False) -> str:
    """Return the normalised form of ``text``.

    If ``mask_numbers`` is true, every digit run is replaced by a single ``0`` token.
    """
    raise NotImplementedError


def fields_to_text(fields: Mapping[str, str], keys: Sequence[str]) -> str:
    """Concatenate the named fields (in the given order) with a unit-separator character.

    Missing keys are an error; the caller decides what to do with them.
    """
    raise NotImplementedError
