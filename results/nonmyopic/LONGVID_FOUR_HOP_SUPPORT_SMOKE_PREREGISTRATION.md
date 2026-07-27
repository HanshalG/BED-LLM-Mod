# LongVidSearch Four-Hop Semantic-Support Smoke Preregistration

## Status

Frozen after the independent four-hop structural confirmation and before any
model response. This is a serving/mechanism smoke only. It does not access a
LongVid task, caption, hidden evidence set, or scientific endpoint.

## Objective

Test the LLM-native component required by the eventual policy:

- generate a weighted semantic support over plausible multi-clip evidence
  chains from a question;
- regenerate that support after a path-dependent caption observation; and
- produce next searches anchored in information newly visible on that branch.

The model does not reason aloud. Reasoning remains reserved for the naive
baseline.

## Exact Calls

Model: `openai/gpt-5.4` through OpenRouter, nonreasoning, temperature `0`,
ordinary chat transport, maximum 900 output tokens, zero retries.

Exact 10 physical requests and HTTP attempts:

1. one discarded parser preflight;
2. one initial six-particle support on the synthetic four-hop fixture; and
3. eight independent support regenerations, one per synthetic branch caption.

The fixture is public in
`scripts/longvid_four_hop_support_smoke.py`; canonical SHA-256:
`6bf2ccc7952689533b9bc609f53b75d1aa4a793ee233f63520e7cf2292cde7ad`.

## Frozen Grammar

Every response is exactly six lines:

`Hn|weight|anchor|search_query|hypothesis`

- labels are `H1` through `H6` in order;
- weight is an integer `1..100`, with at least two distinct weights;
- anchor is one alphanumeric token;
- queries and hypotheses are nonempty, bounded, and distinct; and
- no preface, fence, blank line, extra line, or pipe inside a field is allowed.

Initial anchors are `QUESTION`. On refresh, an anchor must be copied from the
branch observation, absent from the question and previous query, and present
in that line's next search.

Raw batches are checkpointed before parsing. There is no cleanup, repair,
reparse, reissue, retry, or partial-subset analysis.

## Frozen Gates

All must pass:

- exact 10 physical requests and 10 HTTP attempts;
- zero retries, reasoning tokens, and forced exits;
- all 10 responses parse exactly;
- the preflight is discarded;
- initial support has six distinct queries and hypotheses;
- every branch has at least four of six valid observation anchors;
- all eight branch supports differ from the initial support;
- at least six of eight branch supports are mutually distinct; and
- total adapter cost is at most `$0.20`.

Failure closes this exact interface before policy mechanics. Passing authorizes
only a separately frozen disclosed-development policy mechanics gate, not
reserve-video access or a positive claim.

## Budget

Projected cost: `$0.08`. Hard run cap: `$0.20`. The local project ledger ceiling
is frozen at `$105.31076010921015`, exactly `$8.50` above the pre-smoke ledger.
The authenticated balance is `$33.574042594`; at least `$25` remains protected
through Monday. OpenRouter only; no OatML/Slurm.
