# GuessingGame Path-BED Serving Smoke Failure

Date: 2026-07-29

Status: **failed closed; the exact retrieval interface is closed**.

## Bound Protocol

The run was bound to source manifest SHA-256
`8d1269e197c9cf05865852353b16eeadb36cc8db6231af1d429b7650b263e7b8`.
The implementation and frozen gates were committed and pushed as `f5b8325`
before any response.

GPT-5.4-Mini received five material-only and five function-only histories and
the complete 858-object vocabulary. Each response was required to contain
exactly 32 unique weighted object IDs.

## Formal Failure

All ten requests completed, but six responses contained duplicate object IDs.
The strict local parser therefore failed before any semantic aggregate:

| Metric | Result |
|---|---:|
| accepted requests / expected | 10 / 10 |
| HTTP attempts / expected | 10 / 10 |
| responses with 32 unique IDs | 4 / 10 |
| responses with duplicate IDs | 6 / 10 |
| retries | 0 |
| provider-error retries | 0 |
| reasoning tokens | 0 |
| forced exits | 0 |
| cost | `$0.0401607` |

The six invalid supports contained between one and six repeated rows. All ten
responses otherwise had 32 distinct weight values. No deduplication, backfill,
repair, or reissue was performed.

## Diagnostic Only

A saved-response diagnostic ignored duplicate multiplicity solely to determine
whether a uniqueness-only transport change could have rescued the frozen
conjunction. It could not:

| Frozen semantic quantity | Diagnostic | Required |
|---|---:|---:|
| material target recall | 3 / 5 | >=2 / 5 |
| function target recall | 4 / 5 | >=4 / 5 |
| union target recall | 4 / 5 | 5 / 5 |
| material top-16 recall | 2 / 5 | >=1 / 5 |
| function top-16 recall | 4 / 5 | >=3 / 5 |

One target was absent from both supports, so the exact protocol would still
have failed its external semantic endpoint even if every duplicate had been
removed. This diagnostic is not a formal score and does not reopen the run.

## Consequence

The exact model, prompt, five cases, seed, sparse top-32 representation, and
thresholds are closed. Mechanics, development, and confirmation were not
opened. There is no dense-vector successor, target substitution, threshold
change, or new serving split in this cycle.

## Artifacts

- Public failure:
  `results/nonmyopic/guessinggame_path_bed_serving_smoke/guessinggame-path-serving-20260729T020509Z/FAILURE.json`
- Public failure SHA-256:
  `b524f810c6de3dac03df38edc2f53e20a810c373d2e2e281fc8a256369433697`
- Private raw-response SHA-256:
  `498e2ddb9a04e93372d07506c0cb6a3b01ac4fc83c389bff82cdeae779cee689`

`run.log`, `console.log`, source vocabulary, and private raw responses are not
committed.
