"""Compute results/e1/summary.json from the stored E1 recall stage (no detector run).

The summary holds the LSH recall (recall conditional on true Jaccard at or above the threshold),
recall by true Jaccard bin, recall by edit and size, and the length arithmetic that explains the
threshold. Every number is computed from the planted pairs in recall.json.

Usage:
    python scripts/summarise_e1.py                  # results/e1
    python scripts/summarise_e1.py --results DIR
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from benchlint.experiments import RESULTS_DIR, write_e1_summary  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """Write summary.json next to the E1 stage files."""
    parser = argparse.ArgumentParser(description="Summarise E1 from its stored recall stage.")
    parser.add_argument("--results", type=Path, default=RESULTS_DIR / "e1")
    args = parser.parse_args(argv)
    summary = write_e1_summary(args.results)
    print(
        f"E1 summary: LSH recall {summary['lsh_recall_at_threshold']['found']}/"
        f"{summary['lsh_recall_at_threshold']['n']} at or above {summary['threshold']}; "
        f"wrote {args.results / 'summary.json'}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
