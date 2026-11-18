# Pre-registration

Status: stub. Committed before any run, so git history shows the order. Written in Phase 3.

## E1. Detector calibration
- Planted exact and near-duplicates (1, 3, 5, 10-word edits) in 2,000 clean AG News items.
- False-positive rate on 10,000 clean random pairs.
- Paraphrase detection is out of scope (no LLM).

## E2. Known-artifact replication
- Hypothesis-only accuracy on SNLI test (train capped at 100k) and MNLI matched dev.
- Compared with literature values, cited as reported values.

## E3. Leakage scan
- Six public benchmarks (see specs/). Counts, rates and up to 5 example pairs each.

## E4. Dogfooding
- benchlint over quantgate ARC/GSM8K subsets, and over passk task instructions.

## Thresholds, n and decision rules
