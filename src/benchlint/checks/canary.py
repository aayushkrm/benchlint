"""Canary strings: insert a GUID canary into a dataset, and grep corpora for it later."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path


def add_canary(src: Path, dst: Path, *, field: str = "canary", guid: str | None = None) -> str:
    """Write ``dst`` as ``src`` with a ``field`` holding a GUID canary. Return the GUID.

    If ``guid`` is None a fresh UUID4 is generated.
    """
    raise NotImplementedError


def scan_canary(paths: Sequence[Path], guid: str) -> list[tuple[Path, int]]:
    """Return ``(path, line_number)`` for every line of every file containing ``guid``."""
    raise NotImplementedError
