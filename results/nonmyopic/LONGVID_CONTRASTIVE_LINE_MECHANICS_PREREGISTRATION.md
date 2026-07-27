# LongVid Fixed-Line Contrastive Mechanics Preregistration

## Status

Frozen before any model response on these tasks. This is the final LongVid
development interface in this cycle. It cannot rescue either prior transport
failure.

## Motivation And Independence

The six-line support grammar already completed six initial and twelve
real-task refresh responses in the earlier entropy experiment. Its sole
failure was an ASCII-only anchor validator rejecting the observation-grounded
string `€4bn`; line syntax itself did not fail.

This experiment:

- uses four new rows never sent to a model;
- uses contrastive one-step/four-step belief ranking, not entropy;
- parses strict fixed lines rather than JSON;
- validates Unicode anchors with the retrieval tokenizer; and
- keeps the same hidden endpoint discipline and efficacy thresholds.

No prior malformed output is reparsed or reused.

## Frozen Tasks

Bind the official four-hop structural confirmation SHA-256
`895e448c047ca7afa924393da0a4f637a21735c8a770336fa17db0a72fdd4081`.
Exclude all fourteen prior model rows.

| Row | Candidate root indices |
|---:|---|
| 2989 | 0, 17 |
| 549 | 2, 6 |
| 2642 | 4, 0 |
| 2345 | 11, 1 |

Layout seed is `270746`; layout SHA-256 is
`77bd6d1c93c84e2968502541f6a087dfd4257b8ee0ee710845504ca36c1c5909`.

The model never receives greedy/oracle labels, answers, evidence IDs,
coverage, or endpoint-derived values.

## Policy

GPT-5.4 runs through OpenRouter ordinary chat, nonreasoning, temperature `0`,
concurrency `8`, and zero retries.

Each support is exactly six lines:

```text
Hn|weight|anchor|query|hypothesis
```

Each rank is exactly one line:

```text
choice|confidence|unresolved_need
```

The LLM generates an initial support, regenerates it after every retrieved
caption, and chooses the highest-weight next query. A compute-matched myopic
scorer receives only first-step supports/queries; the final scorer receives
all four support/query states. Neither scorer sees raw captions.

Exactly `44` calls are allowed: four initial supports, 32 refreshes, four
one-step ranks, and four four-step ranks.

## Delayed Endpoint And Gates

Necessary-clip IDs remain unloaded until all responses parse, all eight paths
complete, and all choices freeze. No call occurs afterward.

All gates are conjunctive:

- exactly `44` requests and HTTP attempts;
- zero retries, reasoning tokens, and forced exits;
- all outputs parse exactly; no extraction, repair, or reissue;
- all eight paths contain four distinct captions;
- at least `28/32` refreshed supports change;
- all `32` refreshes contain at least four grounded anchor/query pairs;
- at least three tasks have different realized candidate coverage;
- final pairwise accuracy is at least `.75` and strictly above immediate;
- at least two policy choices change and at least one change improves coverage;
- final selected coverage exceeds immediate by at least two clips;
- final selected coverage strictly exceeds seeded random; and
- cost is at most `$0.75`.

Failure closes LongVid contrastive development for this cycle. No remaining
two strict rows, subset, threshold change, model swap, repair, or rerun may be
used. Passage alone authorizes a separately frozen untouched-reserve
confirmation.

## Budget

Projected cost is `$0.40`; cap is `$0.75`. Recheck the authenticated balance
before launch, use no more than the conservative `$33.288895094` remainder,
and retain at least `$25` through Monday, 3 August 2026. OpenRouter only; no
OatML, Slurm, or cluster use.
