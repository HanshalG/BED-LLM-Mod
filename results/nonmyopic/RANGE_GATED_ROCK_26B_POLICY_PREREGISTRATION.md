# Range-Gated RockSample[7,8] 26B Policy Preregistration

## Interface

Gemma 4 26B A4B proposes four named three-action plans from the exact current
position, rock coordinates, per-rock posterior marginals, and the frozen sensor law.
The response contains action strings, not menu indices. Exactly two plans must begin
with distinct movement actions and two with distinct check actions. Plans are checked
for dynamic legality and deduplicated, with one bounded correction attempt.

The model is non-thinking at temperature zero with a 256-token output cap. It is
called once per belief cell. Exact code computes every posterior and plan EIG; there
are no rollout, scoring, or belief-update LLM calls. The model never sees the truth,
EIG values, the selected plan, or control scores.

## S0 Serving Gate

- Model: `google/gemma-4-26B-A4B-it`, direct vLLM on one A100.
- Fresh seed 24145; ten distinct posterior cells on standard RockSample[7,8].
- K4 named plans, horizon three, one bounded validation retry.
- Pass requires ten complete legal distinct plan sets, the exact 2-move/2-check root
  mix, zero reasoning tokens, zero forced exits, and exactly ten accepted logical
  cells. Proposal scores are quarantined for S0.

An S0 failure permits no format repair in this line because the named schema has
already passed deterministic parser and serving-shape tests. A pass authorizes only
the separately frozen S1 below.

## S1 Proposal-Quality Gate

- Fresh seed 24146; sixteen distinct start-position posterior cells formed from
  zero to two weak remote observations that exclude rock 5.
- Every cell must independently be a strict exact h3-over-h2 opportunity: exhaustive
  d3 starts with movement and exhaustive d2 starts with a check.
- LLM K4 h3 plans are scored exactly.
- Matched random uses four legal h3 plans with the same two-move/two-check root mix.
- Shared-plan d2 truncates the exact same LLM plans to two actions, selects under d2,
  then evaluates that selected full plan under the common h3 score.
- Strong d2 selects the exhaustive adaptive d2 root, then receives the best exhaustive
  h3 open-loop continuation under that fixed root.
- Exhaustive open-loop h3 supplies the opportunity ceiling.
- 5,000 paired bootstrap replicates.

All five frozen quality conditions must pass:

1. LLM-minus-matched-random mean plan value has a strictly positive 95% lower bound.
2. LLM-h3-minus-shared-plan-d2 has a strictly positive 95% lower bound.
3. LLM-h3-minus-strong-d2-root has a strictly positive 95% lower bound.
4. The selected LLM root matches the exhaustive h3 route root in at least 75% of cells.
5. Mean recovery of the exhaustive h3-over-strong-d2 opportunity is at least 0.60.

All mechanics, usage accounting, exact scoring, and no-reasoning conditions must also
pass. Failure stops the range-gated LLM policy line. A pass authorizes only a separately
preregistered paired trajectory confirmation; S1 cells cannot enter that endpoint.

## Scope

The exact structural qualification is already banked under fresh seed 24141. This
protocol tests whether a bounded semantic proposal prior can expose the load-bearing
three-step route to the exact verifier. It does not train a policy, use LLM rollouts,
or compare thinking against non-thinking; thinking remains a separate naive baseline.
