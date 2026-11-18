"""Bundled SYNTHETIC demo dataset with planted leaks (for ``benchlint demo``).

Everything produced here is synthetic and must be labelled SYNTHETIC in any output.
"""

from __future__ import annotations

from pathlib import Path

from benchlint.demo_data.generate import DemoBundle, build_demo
from benchlint.demo_data.manifest import (
    SYNTHETIC_NAME,
    DemoManifest,
    LeakRow,
    PlantedLeak,
    compare_planted,
)
from benchlint.schema import DatasetSpec

__all__ = [
    "SYNTHETIC_NAME",
    "DemoBundle",
    "DemoManifest",
    "LeakRow",
    "PlantedLeak",
    "build_demo",
    "build_demo_dataset",
    "compare_planted",
]


def build_demo_dataset(out_dir: Path, *, seed: int = 0) -> DatasetSpec:
    """Write the synthetic dataset with planted exact, near and shortcut leaks to ``out_dir``.

    Returns the ``DatasetSpec`` that points at the written files. Its paths are relative to
    ``out_dir``, so load it with ``base_dir=out_dir``. Use ``build_demo`` to also get the
    reference corpus, the audit spec and the manifest.
    """
    return build_demo(out_dir, seed=seed).spec.dataset
