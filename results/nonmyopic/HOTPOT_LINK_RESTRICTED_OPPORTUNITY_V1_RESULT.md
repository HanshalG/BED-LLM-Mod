# HotpotQA Link-Restricted Opportunity V1 Result

## Decision

The zero-call V1 opportunity screen fails before qualification counts because
one official row references an out-of-range supporting-fact sentence.

The mechanics 100 and development 500 endpoint rows were atomically
materialized before row iteration, so those two splits are now considered
open for mechanics and development. No model was called and no selected task,
action graph, policy score, or endpoint aggregate was produced.

## Failure

The frozen source qualification raised:

```text
ValueError: supporting fact references an invalid sentence
```

The V1 screen did not preregister malformed-row exclusion, so the exception is
not silently converted into a nonqualifying row and V1 is closed.

## Data Boundary

- Metadata-only manifest: passed
- Mechanics endpoint rows materialized: `100`
- Development endpoint rows materialized: `500`
- Confirmation endpoint rows materialized: `0`
- Retained-holdout endpoint rows materialized: `0`
- Model calls / cost: `0 / $0`

No aggregate opportunity result exists.

## Admissible Successor

A narrow V2 source-validity amendment may freeze before rerunning:

- catch only `ValueError` raised by the existing `qualification(row)`;
- record the row as malformed and nonqualifying;
- preserve every split, structural definition, count threshold, and ordering;
- report malformed counts explicitly.

This is an ETL validity rule, not a task or efficacy threshold change.
Confirmation and retained holdout remain sealed.
