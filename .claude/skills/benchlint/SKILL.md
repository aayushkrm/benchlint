---
name: benchlint
description: Audit an ML benchmark dataset for train/test leakage, exact and near duplicates, label conflicts, n-gram contamination and shortcut features with the benchlint CLI. Use when asked whether a benchmark or eval set is clean, or before trusting a score.
---

# benchlint

Status: scaffold. The commands below are the planned interface; check `benchlint --help`
for what is implemented in the installed version.

## When to use

- Before reporting a score on a public benchmark, check for train/test leakage.
- Before adding an eval set, check it for near-duplicates against training data.
- When a single field seems to predict the label, run the shortcut check.

## Commands

```
benchlint audit specs/<name>.yaml --out reports/ --fail-on high
benchlint check exact specs/<name>.yaml
benchlint check near specs/<name>.yaml --threshold 0.8
benchlint check ngram specs/<eval>.yaml --reference <source>.yaml --n 13
benchlint check shortcut specs/<name>.yaml
benchlint canary add in.jsonl out.jsonl
benchlint canary scan <GUID> <paths...>
benchlint demo
```

Add `--json` to any command for machine-readable output.

## Exit codes

- 0: pass.
- 1: a check failed (a finding at or above `--fail-on`, or a canary hit).
- 2: usage or runtime error.

## Rules for interpreting results

- Report counts with rates and denominators, and say which splits were compared.
- The E1 calibration table (planted-leak recall, false-positive rate) states the limits.
  Quote it with any finding.
- Paraphrase leakage is out of scope: there is no LLM or embedding model.
- Shortcut flags are relative to the majority and chance baselines and a margin of 10 points.
  A flag is a prompt to investigate, not proof of a shortcut.
- Literature values (for example the SNLI hypothesis-only result) are cited as reported
  values, never as this tool's results.
