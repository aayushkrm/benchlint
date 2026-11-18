"""Command-line interface (typer).

Exit codes (DECISION.md section 2): 0 = pass, 1 = check failed (finding at or above
``--fail-on``), 2 = usage or runtime error. Every command gets ``--json`` output.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(
    help="benchlint: find leakage, near-duplicates, label conflicts and shortcuts in ML data.",
    no_args_is_help=True,
    add_completion=False,
)
check_app = typer.Typer(help="Run one check on a dataset spec.", no_args_is_help=True)
canary_app = typer.Typer(help="Add or scan GUID canary strings.", no_args_is_help=True)
app.add_typer(check_app, name="check")
app.add_typer(canary_app, name="canary")


@app.command()
def audit(
    spec: Annotated[
        Path, typer.Argument(help="DatasetSpec YAML file (for example specs/gsm8k.yaml).")
    ],
    out: Annotated[Path, typer.Option("--out", help="Directory for report files.")] = Path(
        "reports"
    ),
    fail_on: Annotated[
        str,
        typer.Option(
            "--fail-on", help="Exit 1 if a finding is at or above this: info|low|medium|high."
        ),
    ] = "high",
    json_output: Annotated[
        bool, typer.Option("--json", help="Print the report as JSON to stdout.")
    ] = False,
) -> None:
    """Run every enabled check on the dataset and write JSON, Markdown and HTML reports."""
    raise NotImplementedError


@check_app.command("exact")
def check_exact(
    spec: Annotated[Path, typer.Argument(help="DatasetSpec YAML file.")],
    json_output: Annotated[bool, typer.Option("--json", help="Print findings as JSON.")] = False,
) -> None:
    """Exact duplicates within and across splits."""
    raise NotImplementedError


@check_app.command("near")
def check_near(
    spec: Annotated[Path, typer.Argument(help="DatasetSpec YAML file.")],
    threshold: Annotated[float, typer.Option("--threshold", help="Jaccard threshold.")] = 0.8,
    json_output: Annotated[bool, typer.Option("--json", help="Print findings as JSON.")] = False,
) -> None:
    """MinHash LSH near-duplicates with exact Jaccard verification."""
    raise NotImplementedError


@check_app.command("ngram")
def check_ngram(
    spec: Annotated[Path, typer.Argument(help="DatasetSpec YAML file for the eval set.")],
    reference: Annotated[
        str,
        typer.Option("--reference", help="Reference corpus as a SourceSpec YAML file."),
    ],
    n: Annotated[int, typer.Option("--n", help="n-gram length.")] = 13,
    json_output: Annotated[bool, typer.Option("--json", help="Print findings as JSON.")] = False,
) -> None:
    """13-gram overlap between eval items and a reference corpus (decontamination)."""
    raise NotImplementedError


@check_app.command("shortcut")
def check_shortcut(
    spec: Annotated[Path, typer.Argument(help="DatasetSpec YAML file.")],
    json_output: Annotated[bool, typer.Option("--json", help="Print results as JSON.")] = False,
) -> None:
    """Partial-input baselines (single field, length, numeric scan) against majority and chance."""
    raise NotImplementedError


@canary_app.command("add")
def canary_add(
    src: Annotated[Path, typer.Argument(help="Input JSONL, CSV or parquet file.")],
    dst: Annotated[Path, typer.Argument(help="Output file with the canary field added.")],
    field: Annotated[str, typer.Option("--field", help="Name of the canary field.")] = "canary",
    json_output: Annotated[bool, typer.Option("--json", help="Print the GUID as JSON.")] = False,
) -> None:
    """Insert a fresh GUID canary into a dataset file and print the GUID."""
    raise NotImplementedError


@canary_app.command("scan")
def canary_scan(
    guid: Annotated[str, typer.Argument(help="Canary GUID to search for.")],
    paths: Annotated[list[Path], typer.Argument(help="Corpus files or directories to grep.")],
    json_output: Annotated[bool, typer.Option("--json", help="Print hits as JSON.")] = False,
) -> None:
    """Grep corpora for a canary GUID. Exit 1 if any hit is found."""
    raise NotImplementedError


@app.command()
def demo(
    out: Annotated[Path, typer.Option("--out", help="Directory for demo output.")] = Path(
        "demo_out"
    ),
    json_output: Annotated[bool, typer.Option("--json", help="Print the report as JSON.")] = False,
) -> None:
    """Audit the bundled SYNTHETIC dataset with planted leaks. Runs in under 30 s, no downloads."""
    raise NotImplementedError


if __name__ == "__main__":
    app()
