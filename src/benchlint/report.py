"""Report writers: JSON, Markdown and static HTML (SARIF is a stretch goal)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from benchlint.schema import AuditReport


def render_markdown(report: AuditReport) -> str:
    """Render ``report`` as Markdown. Synthetic runs must carry a SYNTHETIC label."""
    raise NotImplementedError


def render_html(report: AuditReport) -> str:
    """Render ``report`` as a self-contained static HTML page (Jinja2 template)."""
    raise NotImplementedError


def write_reports(
    report: AuditReport,
    out_dir: Path,
    formats: Sequence[str] = ("json", "md", "html"),
) -> list[Path]:
    """Write one file per requested format into ``out_dir`` and return the paths written."""
    raise NotImplementedError


def exceeds_fail_on(report: AuditReport, fail_on: str) -> bool:
    """True if any finding or flagged shortcut is at or above the ``fail_on`` severity.

    ``fail_on`` is one of ``info``, ``low``, ``medium``, ``high``. Drives exit code 1.
    """
    raise NotImplementedError
