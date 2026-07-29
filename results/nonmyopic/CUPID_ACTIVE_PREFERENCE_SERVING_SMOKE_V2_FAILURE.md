# CUPID Active-Preference Serving Smoke V2 Failure

Status: **failed closed; CUPID serving route closed**.

Date: 2026-07-29

Public failure:
`results/nonmyopic/cupid_active_preference_serving_smoke_v2/cupid-active-preference-serving-v2-20260729T004432Z/SERVING_FAILURE.json`.
SHA-256:
`4092d459794011a756ae47d4043f1cb553d94203940fce0b1b93d76ea63490ab`.

Private raw-response SHA-256:
`4a006d991df9536903af9fb50b5092be317120ecda3229091b11af1030cefa02`.

## Execution

V2 made exactly five `openai/gpt-5.4-mini` planner calls followed by five
`google/gemini-2.5-flash` target calls on the five frozen, V1-disjoint cases:

- accepted requests / HTTP attempts: `10 / 10`;
- retries and provider-error retries: `0 / 0`;
- reasoning tokens and forced exits: `0 / 0`;
- prompt / completion tokens: `12,656 / 3,825`; and
- reported cost: `$0.02557305`.

The planner received no hidden preference or checklist. The target phase
accessed the five released hidden preferences and generated questions, but no
checklist. No policy endpoint or holdout row was accessed.

## Exact Failure

All five planner responses parsed under the V2 integer-array representation.
Four target responses also returned six valid integer bits. For the remaining
contrastive case, Gemini returned:

```text
[1, null, null, null, null, 0]
```

despite the structured item schema requiring integer `0` or `1`. The strict
target parser rejected the response. No normalization, imputation, retry,
reissue, or target substitution is allowed.

## Saved-Response Diagnostics

The already-paid planner responses show that the frozen opportunity gates
would fail independently of the target parser:

| Case type | Unique questions | Unique hypotheses | Unique signatures | Minimum partition side | Mean entropy (nats) |
|---|---:|---:|---:|---:|---:|
| consistent | 6 | 12 | 9 | 4 | .6602 |
| consistent | 6 | 12 | 10 | 3 | .6189 |
| contrastive | 6 | 12 | 7 | 2 | .5879 |
| contrastive | 6 | 12 | 10 | 2 | .6003 |
| changing | 6 | 12 | 7 | 1 | .4837 |

The frozen minimum was eight unique signatures for every case and a minority
side of at least two for every question. The third and fifth cases miss the
signature gate, and the changing case misses the partition gate.

Among the four valid target responses, exact generated-signature coverage is
`2/4`; the other two are each one bit from the nearest hypothesis. Even if the
invalid fifth response had been exactly covered, the maximum possible total
would be `3/5`, below the frozen `4/5` gate. Thus target coverage fails
decisively without interpreting or repairing the invalid response.

## Decision

CUPID serving is closed for this project. There is no V3, representation
change, parser relaxation, model swap, prompt revision, threshold change,
row removal, or paid continuation. The 60-row holdout remains untouched.

The source remains a plausible active-preference benchmark, but this exact
open-support/binary-interview apparatus does not provide sufficiently reliable
support coverage for a non-myopic policy experiment.
