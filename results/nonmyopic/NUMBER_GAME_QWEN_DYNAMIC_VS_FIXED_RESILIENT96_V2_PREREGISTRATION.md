# Number Game Qwen Dynamic-vs-Fixed Resilient-96 V2 Preregistration

Date frozen: 2026-07-30, after the V1 smoke evaluator failure and before any
V2 smoke or scientific seed is opened.

## Amendment Boundary

V1 opened no scientific seed. Its ten non-scientific smoke responses are not
reconstructed or reused. V2 changes only the smoke evaluator's parser
contract and uses entirely fresh smoke and scientific seeds.

The V1 gate expected the pooled item-isolated parser's `codec_mode` field.
The validator path calls the base strict parser, which raises unless it
receives one strict JSON object with exactly 24 hypothesis items. Its returned
diagnostic instead contains:

- `raw_count == 24`;
- `valid_unique_count`, equal to the parsed support size; and
- exactly the five rejection counters `wrong_fields`, `invalid_name`,
  `invalid_expression`, `inconsistent`, and `duplicate_extension`.

V2 freezes those fields as its parser gate. A synthetic end-to-end smoke must
pass this exact contract and public/private serialization before any paid V2
request.

## Fresh Serving Gate

Run exactly ten Gemini 2.5 Flash validation-support requests on fresh
non-scientific seeds `85000..85009`. Preserve the provider-error fallback
schedule from V1: original seed, then `+10,000,000`, then `+20,000,000`, with
each transition occurring only after original plus four identical-seed
zero-cost provider-error responses.

All V1 transport, support-size, reasoning, forced-exit, fallback-count, and
`$0.04` budget gates remain unchanged. The only amendment is the corrected
base-parser diagnostic gate above. A full pass is hash-bound into the
scientific runner; any failure closes V2.

## Fresh Scientific Experiment

- 96 trees, seeds `80000..80095`.
- Qwen 3.7 Plus non-reasoning planning with two independent generations per
  planning history and the unchanged `1,000,000` second-draw offset.
- Gemini 2.5 Flash target seeds `81000..81095`.
- Sixteen Gemini validation supports per tree, seeds `82000..83535`.
- 20,000 paired tree-bootstrap samples, seed `84000`.

The strict grammar, retained rejuvenation, candidate roots, three-query
horizon, 33 exact canonical concepts, equal weighting, fallback schedule,
scientific gates, mechanics thresholds, exact `11,040` accepted requests,
`$15.75` cap, and `$16.50` starting-balance gate are identical to V1.

Dynamic support versus fixed support must have at least 48 root changes, at
least 3% mean Brier reduction, a paired interval below zero, and more wins
than losses. Dynamic depth three versus myopic must have at least 8% mean
Brier reduction, a paired interval below zero, and at least 60 wins.

No prior partial tree, smoke response, tree exclusion, continuation, endpoint
change, model swap, or unfrozen seed replacement is permitted.
