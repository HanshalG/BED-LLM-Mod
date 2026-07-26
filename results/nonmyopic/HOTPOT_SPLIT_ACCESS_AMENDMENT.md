# HotpotQA Split Access Amendment

Date: 2026-07-26

## Incident

During a zero-call design audit for a possible future-uplift policy, an inline
local analysis loaded the complete frozen HotpotQA distractor-validation
Parquet table with:

```python
table = pyarrow.parquet.read_table(path)
rows = table.to_pylist()
```

The code subsequently filtered all reported aggregates and examples to the
already-open 500-ID opportunity split. It printed no development or holdout
question, answer, context, supporting-fact, ID, or aggregate. Nevertheless,
the full row conversion materially loaded endpoint-bearing columns for all
7,405 records into the analysis process before the ID filter.

## Decision

The 100-ID development split and 5,318-ID holdout split are permanently
quarantined from fresh or confirmatory use in this project. They may only be
described as opened development data. No HotpotQA result after this amendment
may claim untouched-split evidence from the frozen seed-24350 split.

The previously banked 500-ID opportunity audit and one-task causal smoke are
unchanged because their records were already open. The proposed uplift-policy
run is canceled before preregistration, model construction, prompts, responses,
or endpoints.

## Accounting

- OpenRouter requests: `0`
- OpenRouter cost: `$0`
- OatML or cluster use: none
- Files produced by the inline analysis: none
- Source Parquet SHA-256:
  `c20b638ca82b21d04fe12e14ff417ad05153d4d215a65de54497fca4e972f7c6`

This amendment is frozen before selecting or running a replacement experiment.
