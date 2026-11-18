---
name: benchlint
description: Audit an ML benchmark dataset for train/test leakage, exact and near duplicates, label conflicts, n-gram contamination and shortcut features with the benchlint CLI. Use when asked whether a benchmark or eval set is clean, or before trusting a score.
---

# benchlint

benchlint audits a dataset spec and writes JSON, Markdown and HTML reports. It runs offline on
local files and Hugging Face sources. Start with `benchlint demo` to see the output on bundled
SYNTHETIC data.

## When to use

- Before reporting a score on a public benchmark, audit its train and test splits for leakage.
- Before adding an eval set, check it for near-duplicates against the training data.
- When one field seems to predict the label, run the shortcut check.
- When an eval set may be quoted in a reference corpus, run the n-gram check.

## Commands

```
benchlint demo --out demo_out                       # SYNTHETIC data with planted leaks, under 30 s
benchlint audit specs/gsm8k.yaml --out reports --fail-on high
benchlint audit a.yaml b.yaml --out reports         # one subfolder per spec
benchlint check exact specs/gsm8k.yaml
benchlint check near specs/gsm8k.yaml --threshold 0.8
benchlint check ngram specs/gsm8k.yaml --reference refs/web.yaml --n 13
benchlint check shortcut specs/gsm8k.yaml
benchlint canary add in.jsonl out.jsonl --field canary     # prints the GUID
benchlint canary scan <GUID> <paths...>
```

Every command accepts `--json`. With `--json`, stdout carries only the JSON (status lines go to
stderr). On a usage or runtime error with `--json`, stdout carries `{"error": ..., "exit_code": 2}`. `audit --json` prints an AuditReport. The `check` commands print a JSON list of
findings (or shortcut results). `canary scan --json` prints a list of `{path, line}` hits.

Spec files take two layouts. An AuditSpec has a `dataset:` mapping and an optional `checks:`
mapping. A bare DatasetSpec (as in `specs/*.yaml`) is audited with the default checks.

## Exit codes

- 0: pass. No finding with matches at or above `--fail-on`.
- 1: a finding with count > 0 is at or above `--fail-on`, or a shortcut was flagged (treated
  as severity high), or a canary hit was found.
- 2: usage or runtime error (bad option, missing file, invalid spec, check not implemented).

`--fail-on` takes `info`, `low`, `medium`, `high` or `none` (default `high`; `none` never exits 1 for findings). `benchlint demo` exits 1
on purpose: the planted leaks are the expected findings.

## Outputs of `audit`

`--out DIR` gets three files: `report.json` (machine-readable), `report.md` and `report.html`
(static, no network, light and dark). A report marked SYNTHETIC in its dataset name carries a
banner in every format.

## Reading the JSON report

The JSON validates against `schemas/audit_report.schema.json` in this repository (generated from
the `AuditReport` model in `src/benchlint/schema.py`). Top-level keys:

- `schema_version`: always `"1"`. Refuse other values.
- `dataset`, `spec`: the dataset name and its DatasetSpec.
- `findings`: one object per check result. Read `check`, `severity` (info, low, medium, high),
  `splits`, `count`, `denominator`, `rate`, `examples` (at most a few `ExamplePair`s, texts
  truncated to 200 characters) and `details`.
- `shortcuts`: one object per partial-input baseline. Read `view`, `metric`, `value` with
  `ci_low` and `ci_high`, `majority`, `chance`, `n_train`, `n_test`, `p_value` and `flagged`.
- `env_info`: provenance (git SHA, time, CPU, OS, Python, package versions). It also records
  `checks` (the configuration used) and `data_splits` (rows read and kept per split).
- `runtime_s`: wall-clock seconds for the audit.

A finding with `count` 0 did not match anything, even when its severity is not info.

## Rules for interpreting results

- Quote counts with their denominators and the splits compared. Rates in the report carry a
  95% Wilson interval; quote it with the rate.
- The calibration limits apply to every finding. Near-duplicate detection uses MinHash LSH,
  which is probabilistic: a pair just above the threshold can be missed. Paraphrase leakage is
  out of scope (there is no LLM or embedding model).
- A shortcut flag means the lower confidence bound exceeds the majority baseline by the margin
  (10 points by default). It is a prompt to investigate, not proof of a shortcut.
- Literature values (for example a published hypothesis-only result) are cited as reported
  values, never as this tool's results.
- Output from `demo` is SYNTHETIC. Never report it as a result on a real dataset.
