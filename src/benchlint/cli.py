"""Command-line interface (typer).

Exit codes (DECISION.md section 2): 0 = pass, 1 = check failed (finding at or above
``--fail-on``), 2 = usage or runtime error. Every command gets ``--json`` output.
"""

from __future__ import annotations

import functools
import json
import time
from collections.abc import Callable, Iterator, Sequence
from contextvars import ContextVar
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, NoReturn, ParamSpec

import typer
import yaml

from benchlint.checks.canary import add_canary, scan_canary
from benchlint.checks.exact import find_exact_duplicates
from benchlint.checks.labels import find_label_conflicts
from benchlint.checks.near import find_near_duplicates
from benchlint.checks.ngram import find_ngram_overlap
from benchlint.checks.shortcut import run_shortcut_checks
from benchlint.demo_data import build_demo, compare_planted
from benchlint.io import load_dataset, load_dataset_detailed
from benchlint.report import (
    FAIL_ON_NONE,
    SEVERITIES,
    annotate_intervals,
    collect_env_info,
    exceeds_fail_on,
    findings_exceed,
    render_json,
    wilson_interval,
    write_reports,
)
from benchlint.schema import (
    AuditReport,
    AuditSpec,
    DatasetSpec,
    Finding,
    NearConfig,
    NgramConfig,
    Record,
    ShortcutResult,
    SourceSpec,
)

EXIT_PASS = 0
EXIT_FINDING = 1
EXIT_ERROR = 2
DEFAULT_OUT = Path("reports")
REFERENCE_TEXT_FIELD = "text"

app = typer.Typer(
    help="benchlint: find leakage, near-duplicates, label conflicts and shortcuts in ML data.",
    no_args_is_help=True,
    add_completion=False,
)
check_app = typer.Typer(help="Run one check on a dataset spec.", no_args_is_help=True)
canary_app = typer.Typer(help="Add or scan GUID canary strings.", no_args_is_help=True)
app.add_typer(check_app, name="check")
app.add_typer(canary_app, name="canary")

P = ParamSpec("P")


_JSON_OUTPUT: ContextVar[bool] = ContextVar("benchlint_json_output", default=False)


def _fail(message: str) -> NoReturn:
    """Report a usage or runtime error and exit with code 2.

    With ``--json`` the error goes to stdout as ``{"error": ..., "exit_code": 2}``, the same
    shape as passk and quantgate, so a machine reader always gets one JSON object on stdout.
    Otherwise the message goes to stderr as ``error: ...``.
    """
    if _JSON_OUTPUT.get():
        typer.echo(json.dumps({"error": message, "exit_code": EXIT_ERROR}, indent=2))
    else:
        typer.echo(f"error: {message}", err=True)
    raise typer.Exit(EXIT_ERROR)


def _guard(fn: Callable[P, None]) -> Callable[P, None]:
    """Map any unexpected exception to a message and exit code 2 (runtime error).

    Commands that take ``--json`` (a ``json_output`` argument) report their errors as JSON.
    """

    @functools.wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> None:
        token = _JSON_OUTPUT.set(bool(kwargs.get("json_output", False)))
        try:
            fn(*args, **kwargs)
        except typer.Exit:
            raise
        except Exception as exc:
            _fail(f"{type(exc).__name__}: {exc}")
        finally:
            _JSON_OUTPUT.reset(token)

    return wrapper


def _check_fail_on(fail_on: str) -> None:
    """Reject an unknown severity as a usage error. ``none`` disables the finding exit code."""
    allowed = (*SEVERITIES, FAIL_ON_NONE)
    if fail_on not in allowed:
        _fail(f"--fail-on must be one of {', '.join(allowed)}, got {fail_on!r}")


def load_audit_spec(path: Path) -> AuditSpec:
    """Load an AuditSpec YAML (``dataset:`` plus optional ``checks:``).

    A bare DatasetSpec YAML (as in ``specs/*.yaml``, with no ``dataset:`` key) is also accepted
    and audited with the default checks.
    """
    if not path.is_file():
        raise FileNotFoundError(f"spec file not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a mapping at the top level")
    if "dataset" in raw:
        return AuditSpec.model_validate(raw)
    return AuditSpec(dataset=DatasetSpec.model_validate(raw))


def _load(path: Path) -> tuple[AuditSpec, Path]:
    """Load a spec and return it with its directory (the base for relative paths)."""
    return load_audit_spec(path), path.resolve().parent


def _pick_splits(splits: dict[str, list[Record]]) -> tuple[str | None, str | None]:
    """Name of the train and test split. Uses ``train`` and ``test`` (or ``validation``) when
    present, otherwise the leading and trailing split. ``(None, None)`` for a single split.
    """
    names = list(splits)
    if len(names) < 2:
        return (None, None)
    train = "train" if "train" in names else names[0]
    if "test" in names:
        test = "test"
    elif "validation" in names:
        test = "validation"
    else:
        test = names[-1]
    if test == train:
        return (None, None)
    return (train, test)


def _all_records(splits: dict[str, list[Record]]) -> list[Record]:
    return [rec for recs in splits.values() for rec in recs]


def _eval_records(splits: dict[str, list[Record]], test_split: str | None) -> list[Record]:
    """Eval items: the test split when there is one, otherwise every record."""
    if test_split is not None:
        return list(splits[test_split])
    return _all_records(splits)


def _reference_texts(
    sources: Sequence[SourceSpec], base_dir: Path, fields: Sequence[str] = (REFERENCE_TEXT_FIELD,)
) -> Iterator[str]:
    """Stream the text of every reference record: ``fields`` joined with a space.

    Each reference row must contain every field in ``fields``.
    """
    for src in sources:
        ref_spec = DatasetSpec(
            name="reference",
            source=src,
            text_fields=list(fields),
            label_field=None,
            id_field=None,
            max_rows_per_split=None,
            seed=0,
        )
        for recs in load_dataset(ref_spec, base_dir=base_dir).values():
            for rec in recs:
                yield " ".join(rec.fields[f] for f in fields)


def run_audit(spec: AuditSpec, base_dir: Path) -> AuditReport:
    """Run every enabled check on the dataset and return the AuditReport.

    The near-duplicate check always runs. Label conflicts run when the dataset has a label
    field. The shortcut check runs when there is a train and a test split. The n-gram check
    runs when reference corpora are configured. Canary checks are not run here: they need a
    GUID (see ``benchlint canary scan``).
    """
    start = time.perf_counter()
    cfg = spec.checks
    ds = spec.dataset
    splits, loads = load_dataset_detailed(ds, base_dir=base_dir)
    records = _all_records(splits)
    train_split, test_split = _pick_splits(splits)
    findings: list[Finding] = []
    if cfg.exact:
        findings += find_exact_duplicates(records, ds.text_fields)
    findings += find_near_duplicates(records, ds.text_fields, cfg.near, seed=ds.seed)
    if cfg.labels and ds.label_field is not None:
        findings += find_label_conflicts(records, ds.text_fields, near=cfg.near, seed=ds.seed)
    shortcuts: list[ShortcutResult] = []
    if train_split is not None and test_split is not None and cfg.shortcut.views:
        shortcuts = run_shortcut_checks(
            records,
            cfg.shortcut,
            train_split=train_split,
            test_split=test_split,
            seed=ds.seed,
        )
    if cfg.ngram.reference:
        findings.append(
            find_ngram_overlap(
                _eval_records(splits, test_split),
                _reference_texts(cfg.ngram.reference, base_dir, cfg.ngram.reference_fields),
                ds.text_fields,
                cfg.ngram,
            )
        )
    annotate_intervals(findings)
    env = collect_env_info()
    env["checks"] = cfg.model_dump(mode="json")
    env["data_splits"] = [asdict(x) for x in loads]
    return AuditReport(
        dataset=ds.name,
        spec=ds,
        findings=findings,
        shortcuts=shortcuts,
        env_info=env,
        runtime_s=round(time.perf_counter() - start, 3),
    )


def _finding_line(f: Finding) -> str:
    lo, hi = wilson_interval(f.count, f.denominator)
    rate = 100 * f.rate
    return (
        f"[{f.severity:<6}] {f.check:<22} {f.count}/{f.denominator} ({rate:.1f}%, "
        f"95% CI {100 * lo:.1f}-{100 * hi:.1f}%) splits: {', '.join(f.splits)}"
    )


def _shortcut_line(s: ShortcutResult) -> str:
    flag = "FLAGGED" if s.flagged else "ok"
    return (
        f"[{flag:<7}] shortcut {s.view}: {s.metric} {s.value:.3f} "
        f"[{s.ci_low:.3f}, {s.ci_high:.3f}] vs majority {s.majority:.3f}, chance {s.chance:.3f}"
    )


def _emit_check(
    findings: list[Finding],
    shortcuts: list[ShortcutResult],
    json_output: bool,
    fail_on: str,
) -> None:
    """Print check results (JSON list or text lines) and exit 1 if they reach ``fail_on``."""
    annotate_intervals(findings)
    if json_output:
        typer.echo(
            json.dumps(
                [f.model_dump(mode="json") for f in findings]
                if not shortcuts
                else [s.model_dump(mode="json") for s in shortcuts],
                indent=2,
            )
        )
    else:
        for f in findings:
            typer.echo(_finding_line(f))
        for s in shortcuts:
            typer.echo(_shortcut_line(s))
        if not findings and not shortcuts:
            typer.echo("no results")
    if findings_exceed(findings, shortcuts, fail_on):
        raise typer.Exit(EXIT_FINDING)


def _expand_paths(paths: Sequence[Path]) -> list[Path]:
    """Files as given, and every file under each directory (sorted)."""
    files: list[Path] = []
    for p in paths:
        if p.is_dir():
            files.extend(sorted(q for q in p.rglob("*") if q.is_file()))
        elif p.is_file():
            files.append(p)
        else:
            raise FileNotFoundError(f"path not found: {p}")
    return files


@app.command()
@_guard
def audit(
    specs: Annotated[
        list[Path],
        typer.Argument(help="AuditSpec (or DatasetSpec) YAML files, for example specs/gsm8k.yaml."),
    ],
    out: Annotated[Path, typer.Option("--out", help="Directory for report files.")] = DEFAULT_OUT,
    fail_on: Annotated[
        str,
        typer.Option(
            "--fail-on", help="Exit 1 if a finding is at or above this: info|low|medium|high|none."
        ),
    ] = "high",
    json_output: Annotated[
        bool, typer.Option("--json", help="Print the report as JSON to stdout.")
    ] = False,
) -> None:
    """Run every enabled check on each dataset and write JSON, Markdown and HTML reports.

    With one spec the files go directly into the --out directory. With several, each goes into
    a subfolder named after the spec file. Exit code 1 if any report has a finding at or above
    --fail-on.
    """
    _check_fail_on(fail_on)
    say = functools.partial(typer.echo, err=json_output)
    reports: list[AuditReport] = []
    failed = False
    for path in specs:
        spec, base = _load(path)
        if spec.checks.canary:
            say("note: canary is not run by audit; use `benchlint canary scan` with a GUID")
        report = run_audit(spec, base)
        target = out if len(specs) == 1 else out / path.stem
        written = write_reports(report, target)
        reports.append(report)
        n_findings = sum(1 for f in report.findings if f.count > 0)
        flagged = sum(1 for s in report.shortcuts if s.flagged)
        say(
            f"audit: {report.dataset}: {n_findings} finding(s) with matches, "
            f"{flagged} shortcut(s) flagged"
        )
        for p in written:
            say(f"  wrote {p}")
        if exceeds_fail_on(report, fail_on):
            failed = True
            say(f"  FAIL: a finding or shortcut is at or above {fail_on}")
    if json_output:
        if len(reports) == 1:
            typer.echo(render_json(reports[0]), nl=False)
        else:
            typer.echo(json.dumps([json.loads(render_json(r)) for r in reports], indent=2))
    if failed:
        raise typer.Exit(EXIT_FINDING)


@check_app.command("exact")
@_guard
def check_exact(
    spec: Annotated[Path, typer.Argument(help="DatasetSpec YAML file.")],
    json_output: Annotated[bool, typer.Option("--json", help="Print findings as JSON.")] = False,
    fail_on: Annotated[
        str, typer.Option("--fail-on", help="Exit 1 at or above: info|low|medium|high|none.")
    ] = "high",
) -> None:
    """Exact duplicates within and across splits."""
    _check_fail_on(fail_on)
    audit_spec, base = _load(spec)
    splits = load_dataset(audit_spec.dataset, base_dir=base)
    findings = find_exact_duplicates(_all_records(splits), audit_spec.dataset.text_fields)
    _emit_check(findings, [], json_output, fail_on)


@check_app.command("near")
@_guard
def check_near(
    spec: Annotated[Path, typer.Argument(help="DatasetSpec YAML file.")],
    threshold: Annotated[
        float | None,
        typer.Option("--threshold", help="Jaccard threshold. Default: the spec's value (0.8)."),
    ] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Print findings as JSON.")] = False,
    fail_on: Annotated[
        str, typer.Option("--fail-on", help="Exit 1 at or above: info|low|medium|high|none.")
    ] = "high",
) -> None:
    """MinHash LSH near-duplicates with exact Jaccard verification."""
    _check_fail_on(fail_on)
    audit_spec, base = _load(spec)
    near_cfg: NearConfig = audit_spec.checks.near
    if threshold is not None:
        if not 0.0 < threshold <= 1.0:
            _fail("--threshold must be in (0, 1]")
        near_cfg = near_cfg.model_copy(update={"threshold": threshold})
    splits = load_dataset(audit_spec.dataset, base_dir=base)
    findings = find_near_duplicates(
        _all_records(splits),
        audit_spec.dataset.text_fields,
        near_cfg,
        seed=audit_spec.dataset.seed,
    )
    _emit_check(findings, [], json_output, fail_on)


@check_app.command("ngram")
@_guard
def check_ngram(
    spec: Annotated[Path, typer.Argument(help="DatasetSpec YAML file for the eval set.")],
    reference: Annotated[
        str | None,
        typer.Option(
            "--reference",
            help="Reference corpus as a SourceSpec YAML file. Default: the spec's ngram.reference.",
        ),
    ] = None,
    n: Annotated[
        int | None, typer.Option("--n", help="n-gram length. Default: the spec's value (13).")
    ] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Print findings as JSON.")] = False,
    fail_on: Annotated[
        str, typer.Option("--fail-on", help="Exit 1 at or above: info|low|medium|high|none.")
    ] = "high",
) -> None:
    """13-gram overlap between eval items and a reference corpus (decontamination)."""
    _check_fail_on(fail_on)
    audit_spec, base = _load(spec)
    sources: list[SourceSpec] = list(audit_spec.checks.ngram.reference)
    if reference is not None:
        ref_path = Path(reference)
        if not ref_path.is_file():
            _fail(f"reference spec not found: {ref_path}")
        raw = yaml.safe_load(ref_path.read_text(encoding="utf-8"))
        if isinstance(raw, dict) and "source" in raw:
            raw = raw["source"]
        sources = [SourceSpec.model_validate(raw)]
    if not sources:
        _fail("no reference corpus: pass --reference or set checks.ngram.reference")
    ngram_cfg: NgramConfig = audit_spec.checks.ngram.model_copy(
        update={"reference": sources, **({"n": n} if n is not None else {})}
    )
    splits = load_dataset(audit_spec.dataset, base_dir=base)
    _, test_split = _pick_splits(splits)
    finding = find_ngram_overlap(
        _eval_records(splits, test_split),
        _reference_texts(sources, base, ngram_cfg.reference_fields),
        audit_spec.dataset.text_fields,
        ngram_cfg,
    )
    _emit_check([finding], [], json_output, fail_on)


@check_app.command("shortcut")
@_guard
def check_shortcut(
    spec: Annotated[Path, typer.Argument(help="DatasetSpec YAML file.")],
    json_output: Annotated[bool, typer.Option("--json", help="Print results as JSON.")] = False,
    fail_on: Annotated[
        str, typer.Option("--fail-on", help="Exit 1 at or above: info|low|medium|high|none.")
    ] = "high",
) -> None:
    """Partial-input baselines (single field, length, numeric scan) against majority and chance."""
    _check_fail_on(fail_on)
    audit_spec, base = _load(spec)
    splits = load_dataset(audit_spec.dataset, base_dir=base)
    train_split, test_split = _pick_splits(splits)
    if train_split is None or test_split is None:
        _fail("the shortcut check needs a train and a test split")
    results = run_shortcut_checks(
        _all_records(splits),
        audit_spec.checks.shortcut,
        train_split=train_split,
        test_split=test_split,
        seed=audit_spec.dataset.seed,
    )
    _emit_check([], results, json_output, fail_on)


@canary_app.command("add")
@_guard
def canary_add(
    src: Annotated[Path, typer.Argument(help="Input JSONL, CSV or parquet file.")],
    dst: Annotated[Path, typer.Argument(help="Output file with the canary field added.")],
    field: Annotated[str, typer.Option("--field", help="Name of the canary field.")] = "canary",
    json_output: Annotated[bool, typer.Option("--json", help="Print the GUID as JSON.")] = False,
) -> None:
    """Insert a fresh GUID canary into a dataset file and print the GUID."""
    if not src.is_file():
        _fail(f"input not found: {src}")
    guid = add_canary(src, dst, field=field)
    if json_output:
        typer.echo(json.dumps({"guid": guid, "field": field, "dst": str(dst)}, indent=2))
    else:
        typer.echo(guid)


@canary_app.command("scan")
@_guard
def canary_scan(
    guid: Annotated[str, typer.Argument(help="Canary GUID to search for.")],
    paths: Annotated[list[Path], typer.Argument(help="Corpus files or directories to grep.")],
    json_output: Annotated[bool, typer.Option("--json", help="Print hits as JSON.")] = False,
) -> None:
    """Grep corpora for a canary GUID. Exit 1 if any hit is found."""
    files = _expand_paths(paths)
    hits = scan_canary(files, guid)
    if json_output:
        typer.echo(json.dumps([{"path": str(p), "line": line} for p, line in hits], indent=2))
    else:
        for p, line in hits:
            typer.echo(f"{p}:{line}")
        if not hits:
            typer.echo("no hits")
    if hits:
        raise typer.Exit(EXIT_FINDING)


@app.command()
@_guard
def demo(
    out: Annotated[Path, typer.Option("--out", help="Directory for demo output.")] = Path(
        "demo_out"
    ),
    fail_on: Annotated[
        str, typer.Option("--fail-on", help="Exit 1 at or above: info|low|medium|high|none.")
    ] = "high",
    json_output: Annotated[
        bool, typer.Option("--json", help="Print the AuditReport as JSON to stdout.")
    ] = False,
) -> None:
    """Audit the bundled SYNTHETIC dataset with planted leaks. Runs in under 30 s, no downloads.

    Writes the data, the audit spec, the planted manifest, the reports and
    ``planted_vs_found.json`` into ``--out``. Exit 1 is the expected result for the planted
    leaks (use ``--fail-on`` to change the threshold).
    """
    _check_fail_on(fail_on)
    say = functools.partial(typer.echo, err=json_output)
    bundle = build_demo(out, seed=0)
    report = run_audit(bundle.spec, base_dir=out)
    written = write_reports(report, out)
    rows = compare_planted(bundle.manifest, report)
    (out / "planted_vs_found.json").write_text(
        json.dumps([r.model_dump(mode="json") for r in rows], indent=2) + "\n", encoding="utf-8"
    )
    say("SYNTHETIC demo: generated data with planted leaks (seed 0). No real dataset is used.")
    say("")
    say(f"{'category':<22} {'check':<9} {'planted':>7}  {'found':>7}  status")
    for r in rows:
        found = "n/a" if r.found is None else str(r.found)
        say(f"{r.category:<22} {r.check:<9} {r.planted:>7}  {found:>7}  {r.status} ({r.note})")
    say("")
    n_findings = sum(1 for f in report.findings if f.count > 0)
    flagged = sum(1 for s in report.shortcuts if s.flagged)
    say(f"audit: {n_findings} finding(s) with matches, {flagged} shortcut(s) flagged")
    for p in [*written, out / "planted_vs_found.json", out / "manifest.json"]:
        say(f"  wrote {p}")
    if json_output:
        typer.echo(render_json(report), nl=False)
    if findings_exceed(report.findings, report.shortcuts, fail_on):
        raise typer.Exit(EXIT_FINDING)


if __name__ == "__main__":
    app()
