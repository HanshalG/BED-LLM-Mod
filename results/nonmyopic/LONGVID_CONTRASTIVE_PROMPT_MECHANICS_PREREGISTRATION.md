# LongVid Contrastive Prompt-Only Mechanics Preregistration

## Status

Frozen after the passing synthetic transport smoke and before any model
response on these tasks. This is a disclosed structural-confirmation
development gate, not an untouched reserve result.

## Bound Inputs

- Official LongVidSearch code:
  `4aa5620e06ca1bc5cc7cfce2b3e3ed0a5ae82d4e`.
- QA SHA-256:
  `370711f1299202cd40d559440859476bf3a2d2ca74540dad5226786558ee123b`.
- caption SHA-256:
  `0f2ce94265b7050eaee5de239a760c1a0f0762c1754cbb3bd7b45d00774b6c68`.
- four-hop structural confirmation:
  `895e448c047ca7afa924393da0a4f637a21735c8a770336fa17db0a72fdd4081`.
- passing prompt-only serving smoke:
  `fe332c80f79bc4f8eaff6dbbe2d9cc7831e1dc0b6648a07d8037784ac7890068`.

All ten rows previously sent to a model are excluded. The next four strict
structural-confirmation rows and independently blinded roots are:

| Row | Candidate root indices |
|---:|---|
| 1404 | 2, 0 |
| 2703 | 0, 19 |
| 1867 | 5, 4 |
| 1295 | 4, 0 |

Layout seed is `270744`; layout SHA-256 is
`90090d9912857fb7f4dea4992933e845e4a0f9367504f52215ce9160d3e10ba0`.

## Policy

GPT-5.4 runs through ordinary OpenRouter chat with reasoning disabled,
temperature `0`, concurrency `8`, and zero retries.

For each task:

1. generate six weighted open-world evidence-chain hypotheses;
2. execute each of two fixed blinded root searches;
3. after every caption, regenerate all six hypotheses and next searches;
4. execute the highest-weight next search, excluding prior captions;
5. ask one scorer to compare only the first-step belief/query states; and
6. ask a compute-matched scorer to compare all four belief/query states.

The scorers never see raw captions, answers, evidence IDs, coverage, or
greedy/oracle labels. The LLM-regenerated support is therefore the sole
semantic state passed from observation to policy value.

Exactly `44` prompt-only flat-JSON requests are allowed: four initial
supports, 32 refreshes, four immediate ranks, and four final ranks.

## Delayed Endpoint

Necessary-clip IDs remain unloaded until every response parses, all eight
paths are complete, both policy choices freeze, and a pre-endpoint checkpoint
is written. No call occurs after endpoint access.

## Frozen Gates

All are conjunctive:

- exactly `44` physical requests and HTTP attempts;
- zero retries, reasoning tokens, and forced exits;
- all outputs parse exactly with no extraction, repair, or reissue;
- all eight paths contain four distinct captions;
- at least `28/32` refreshed supports change;
- all `32` refreshes contain at least four grounded anchor/query pairs;
- at least three tasks have different realized candidate coverage;
- final pairwise accuracy is at least `.75`;
- final pairwise accuracy strictly exceeds immediate accuracy;
- immediate and final choices differ on at least two tasks;
- at least one choice change strictly improves coverage;
- final selected coverage exceeds immediate selected coverage by at least two
  necessary clips;
- final selected coverage strictly exceeds seeded-random coverage; and
- adapter cost is at most `$0.75`.

Failure closes this prompt-only contrastive development method. No task
subset, threshold change, alternate model, repair, reissue, or rerun is
allowed. Passage alone authorizes a separately frozen experiment on untouched
four-hop reserve videos.

## Budget

Projected cost is `$0.40`; hard cap is `$0.75`. Use the lower of the live
provider balance and `$33.321552594`, retain at least `$25` through Monday,
3 August 2026, and recheck immediately before launch. OpenRouter only; no
OatML, Slurm, or cluster use.
