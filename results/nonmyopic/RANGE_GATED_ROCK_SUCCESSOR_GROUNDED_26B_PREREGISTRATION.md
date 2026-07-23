# Range-Gated Rock Successor-Grounded 26B Preregistration

Status: frozen after implementation and deterministic tests, before any response from
this model/interface combination.

## Motivation and treatment

The independently audited exact benchmark has a large horizon-three structural gap:
exact d3 beats exact d2 by `+0.46296` entropy-AUC nats over 500 pairs because the rover
must move south twice before checking rock 5 on site. The non-thinking GPT-5.4 Mini
fixed-tail smoke repaired serving but found that route in only `1/10` cells. In the
other nine cells it moved east from `(0,4)` and remotely checked a rock not located at
the resulting coordinate.

This is a separately frozen model-and-representation treatment, not a rerun of the
failed Mini line:

1. The model is `google/gemma-4-26b-a4b-it` with a 4,096-token thinking budget and
   256-token final-answer budget.
2. The four machine-fixed roots, K4 exact verifier, sensor law, marginals, and strict
   two-action-tail JSON grammar are unchanged.
3. Each root slot additionally lists the successor coordinate of every legal second
   action. This is a deterministic transition table, not a utility card: it supplies
   no EIG, plan value, rank, preferred action, selected plan, or on-site label.
4. The model must cross-reference successor coordinates with the separately listed
   rock coordinates and choose the third action. Exact scoring and belief updates make
   no model calls.

## S0 serving and route smoke

- Fresh seed `24179`; ten distinct start-position posterior cells with zero to two
  weak remote observations excluding rock 5.
- OpenRouter, temperature zero, one bounded correction response.

All requirements must pass:

1. Ten cells complete with four dynamically legal fixed-root plans.
2. Exactly ten accepted logical calls, zero forced exits, and zero scoring-time model
   calls. Reasoning tokens are expected and fully logged.
3. `move-SOUTH, move-SOUTH, check-5` is present in at least `8/10` cells.

Any failure stops this model/representation line without a seed, prompt, budget, or
threshold change.

## Conditional S1 proposal-quality gate

S1 runs only after a complete S0 pass.

- Fresh seed `24180`; sixteen distinct strict h3-over-h2 opportunities.
- Matched random samples legal tails under the identical machine-fixed roots.
- Shared-plan d2, strong exhaustive d2 root, exhaustive open-loop h3 ceiling, and
  5,000 paired bootstrap replicates are unchanged.

All requirements must pass:

1. Positive 95% lower bound for LLM-minus-matched-random exact plan value.
2. Positive 95% lower bound for LLM-h3-minus-shared-plan-d2.
3. Positive 95% lower bound for LLM-h3-minus-strong-d2-root.
4. At least 75% exhaustive h3 route-root selection.
5. At least 60% mean recovery of the h3-over-strong-d2 opportunity.
6. Complete mechanics and independent replay, with all reasoning and forced-exit
   accounting retained and no scoring-time model calls.

A pass authorizes only a separately frozen paired trajectory confirmation. Expected
S0 plus S1 cost is below `$0.50`, within the `$1` run cap. Registered project spend is
`$38.51543566 / $110`, leaving `$71.48456434`.
