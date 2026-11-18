# Limitations

What each check can and cannot show. Read before quoting a finding.

## Matching is lexical

- Exact keys, word shingles and 13-grams match surface text. Paraphrases, translations and
  reordered sentences are missed. E1 states the recall on edited copies, so the limit is measured
  for word-level edits only.
- The n-gram check counts items that share at least one 13-gram with the reference. A shared
  boilerplate phrase can match without contamination.

## LSH recall is probabilistic

- Reported pairs are exact (verified with Jaccard). A pair that is truly above 0.8 can still be
  missed by LSH retrieval. The retrieval threshold is 0.7, which lowers the miss rate; it does not
  remove it. E1 gives the measured recall for the planted edits, by edit type, size and text
  length.

## Short texts and the 0.8 threshold

- Word-level edits change up to five 5-gram shingles each. A text of n words has S = n - 4
  shingles, and its copy keeps at least (S - 5k) / (S + 5k) Jaccard after k edits. For a 0.8
  threshold this needs S >= 45k, so one edit needs about 49 words, three edits about 139, five
  about 229 and ten about 454.
- A text shorter than the guaranteed length for k edits (bound table) can fall below 0.8 with k edits even when retrieval works. For one edit that length is about 49 words. E1 reports
  this in two ways: recall conditional on the true Jaccard being at or above the threshold (the
  LSH recall), and recall by true Jaccard bin. Read the overall recall together with those.
- Options for a user who needs short edits caught: a smaller shingle size (each word edit then
  destroys fewer shingles, so Jaccard stays higher, at the cost of more chance matches on common
  phrases), a lower threshold (more false positives, measured by E1's clean-pair
  rate), or a different check. None of these is the default here, and none has been measured on
  other data.

## Thresholds are conventions

- 0.8 is the reporting threshold and 5-word shingles are the default. Other choices change the
  result. Both are configurable, and E1 reports how the default behaves.
- A near-duplicate count depends on the shingle size: short texts have fewer shingles, so one
  edit moves Jaccard more. E1 bins by text length for this reason.

## Shortcut baselines

- A flag depends on the baseline classifier (TF-IDF with logistic regression, a shallow decision
  tree for length, a single numeric feature). A stronger classifier could find more signal. The
  flag says the chosen view beats majority by a margin, not that the field has no other signal.
- Permutation p-values for the numeric scan are not corrected for multiple comparisons.

## Templates and false positives

- Templated text (GSM8K phrasing, ARC stems) shares many shingles. Such pairs can be flagged.
  Each near finding reports its Jaccard histogram and its example pairs so a reader can check.

## Membership inference is out of scope

- Telling whether a closed model memorised an item needs model outputs, and those are weak
  evidence. benchlint does not claim to detect memorisation.

## Data and licences

- Results depend on the dataset versions the Hub served at run time. The spec files name the
  configs and splits checked on 2026-11-19; the Hub can change.
- Some dataset licences are not stated on their cards (see README). benchlint does not
  redistribute the text.

## Experiments

- E1 uses synthetic edits of real text. Its results measure the detector on known edits and say
  nothing about how often benchmarks leak.
- E2 compares with literature values checked against Table 2 of the cited paper (arXiv:1803.02324), not measured here. The classifier differs from the
  one in the cited paper, so the comparison is about shortcut strength, not a replication.
- E4 (dogfooding on the sibling projects) has not been run.
- The E1 and E3 results come from working trees with uncommitted changes. The provenance table in the README and `docs/deviations.md` say which files, and what the recorded commit does and does not reproduce.
