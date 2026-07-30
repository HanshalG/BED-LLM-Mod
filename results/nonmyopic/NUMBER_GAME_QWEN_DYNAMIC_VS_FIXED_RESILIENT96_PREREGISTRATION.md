# Number Game Qwen Dynamic-vs-Fixed Resilient-96 Preregistration

Date frozen: 2026-07-30, after the powered-96 transport failure and before any
serving-smoke or scientific seed in this successor is opened.

## Scope

This is a transport-only successor to the permanently failed powered-96
attempt. It uses entirely new scientific seeds and leaves the policy,
generators, likelihoods, validation model, endpoint, estimands, and efficacy
gates unchanged.

The failed attempt's 13 complete trees and partial fourteenth tree are never
scored or reused. Its status remains `failed_closed`.

## Provider-Error Fallback

The existing seeded adapter makes five total attempts with the identical
payload: one original plus four exponential-backoff retries. The failed run
showed that a seed-specific `finish_reason="error"` can persist across that
identical schedule.

For this successor only:

- after all five responses for one seed are zero-cost provider errors, retry
  the same request with `seed + 10,000,000`;
- if that seed also exhausts five zero-cost provider-error responses, retry
  once with `seed + 20,000,000`;
- then fail closed;
- count each transition to a fallback seed as an additional retry so exact
  transport accounting remains
  `HTTP attempts = accepted requests + retries`;
- record model, original seed, exhausted seed, fallback seed, and fallback
  group for every transition.

The fallback is forbidden for HTTP exceptions that do not already pass the
base adapter's retry policy, paid responses, valid model responses, parser
failures, malformed JSON, low-support outputs, forced exits, or any scientific
outcome. A fallback response is one accepted request, exactly like an
ordinary successful retry.

## Serving Gate

Before scientific execution, make exactly ten fresh Gemini 2.5 Flash
no-history validation-support requests using seeds `79000..79009` through the
fallback adapter.

All gates must pass:

- exactly ten accepted requests;
- exact transport accounting;
- all ten outputs strict JSON;
- every support has at least 16 valid unique extensions;
- zero reasoning tokens and forced exits;
- at most ten fallback transitions;
- provider-error retries do not exceed total retries;
- cost at most `$0.04`.

The smoke contains no scientific tree, target, candidate action, or efficacy
endpoint. A full pass is hash-bound into the scientific runner. Any failure
closes this successor without opening scientific seeds.

## Fresh Scientific Experiment

- 96 fresh trees, seeds `73000..73095`.
- Qwen 3.7 Plus non-reasoning planning with two independent generations at
  each of 49 planning histories and the existing `1,000,000` second-draw
  offset.
- Gemini 2.5 Flash target seeds `74000..74095`.
- Sixteen Gemini validation supports per tree, seeds `75000..76535`.
- Unchanged strict grammar, retained rejuvenation, candidate roots,
  three-query horizon, 33 exact canonical concepts, and equal tree/target
  weighting.
- 20,000 paired tree-bootstrap samples, seed `78000`.

## Frozen Scientific Gates

Dynamic-support depth three versus fixed-initial-support depth three must:

- select different roots on at least `48/96` trees;
- reduce mean Brier by at least `3%`;
- have a paired 95% tree-bootstrap Brier-difference interval below zero; and
- have strictly more Brier wins than losses, excluding exact ties.

Co-required dynamic depth three versus myopic EIG must:

- reduce mean Brier by at least `8%`;
- have a paired 95% tree-bootstrap interval below zero; and
- win at least `60/96` trees.

These are identical to the powered-96 gates frozen before its failure.
Hamming, coverage, depth-two, first-link, prior-run pooling, and meta-analysis
are diagnostic only.

## Mechanics And Budget

- Exactly `11,040` accepted requests and `11,040` parsed provider draws.
- Exactly `4,704` pooled planning histories and `6,336` parse events.
- Exact transport accounting, at most 480 total retries, and at most 96 seed
  fallback transitions.
- Zero reasoning tokens and forced exits; at most 48 item-salvaged draws.
- Every pooled initial support at least 24; every deployed retained
  first/second support at least 12/eight; every validation support at least
  16.
- Cost at most `$15.75`; starting balance at least `$16.50`.

Report once as `passed`, `gated_null`, or `failed_closed`. No continuation,
seed replacement outside the frozen fallback schedule, partial-tree reuse,
tree exclusion, outlier removal, third draw, semantic repair, threshold
change, support weighting change, endpoint change, or model swap is allowed.
