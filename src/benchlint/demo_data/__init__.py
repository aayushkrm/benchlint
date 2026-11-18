"""Bundled SYNTHETIC demo dataset with planted leaks (for ``benchlint demo``).

Everything produced here is synthetic and must be labelled SYNTHETIC in any output.
"""

from __future__ import annotations

from pathlib import Path

from benchlint.schema import DatasetSpec


def build_demo_dataset(out_dir: Path, *, seed: int = 0) -> DatasetSpec:
    """Write the synthetic dataset with planted exact, near and shortcut leaks to ``out_dir``.

    Returns the ``DatasetSpec`` that points at the written files.
    """
    raise NotImplementedError
