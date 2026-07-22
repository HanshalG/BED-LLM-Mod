# Gated Sensor v3 Proposal-Quality Smoke

Registered 2026-07-22 before any indexed-v3 model response. This is an exploratory
interface qualification, not a policy endpoint and not part of a later formal
confidence interval.

## Change Under Test

The task, fixed K4 roots, legal follow-up menus, exact scorer, posterior, horizon,
and controls are unchanged from indexed v2. The v3 prompt additionally supplies,
for each hypothetical root outcome, the exact posterior predicate marginals that
would be available after that branch. It does not supply EIG, an oracle action or
value, hidden truth, or realized future observation. This removes mental Bayesian
arithmetic from the proposer while retaining branch-continuation choice as its job.

## Exploratory Matrix

- Non-thinking direct-vLLM Gemma 4 12B and 26B A4B.
- Four paired trials, eight rounds, K4, seed `24110`, temperature zero.
- Existing indexed-v2 prompt and branch-conditioned indexed-v3 prompt use the same
  model, seed, and fixed roots. These cells may select whether v3 merits a fresh
  formal experiment but cannot support a paper endpoint.

## Qualification Rule

The interface qualifies for a fresh preregistered confirmation only if at least one
model has zero terminal/legality failures and, over its 28 nonterminal h2 states:

- mean exact same-root continuation efficiency is at least `0.90`; and
- continuation efficiency exceeds its matched-random arm by at least `+0.05`.

Continuation efficiency is proposed branch-policy value divided by exact value
after optimizing only the continuation under the identical fixed roots. Root
coverage and proposal/exhaustive-d2 fraction are reported separately. No failed
model is replaced, and no threshold is changed after responses.
