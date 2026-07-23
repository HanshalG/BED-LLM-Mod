# Range-Gated Rock Successor-Grounded Qwen 14B Preregistration

Status: frozen after an endpoint-free JSON serving calibration, before any map, route,
belief, or policy response from Qwen under this treatment.

## Model-selection rationale

The exact Range-Gated Rock benchmark has a registered `+0.46296` entropy-AUC d3-over-d2
gap. Two prior serving smokes are closed:

- non-thinking GPT-5.4 Mini produced legal plans but found the delayed on-site route in
  only `1/10` cells;
- successor-grounded Gemma 4 26B A4B exhausted its thinking budget twice, returned
  `None`, and accepted zero cells.

OpenRouter currently serves `qwen/qwen3-14b`, a 14.8B dual-mode model. Before this
registration, one generic task containing no experimental state requested the JSON
object `{"a":1,"b":1,"c":1,"d":1}`. Qwen returned it exactly after 155 reasoning
tokens with zero forced exits. That calibration selected the serving model only and
did not expose a scientific endpoint.

## Treatment

- Qwen 3 14B with a 4,096-token thinking budget and 256-token final-answer budget.
- The successor-grounded fixed-root interface is unchanged: two geometry roots, two
  exact-d1 check roots, legal second-action successor coordinates, separately listed
  rock coordinates, sensor law, marginals, and strict two-action-tail JSON.
- No EIG, plan value, action rank, preferred tail, selected plan, truth, or prior model
  response is exposed.
- Exact scoring and belief updates make no model calls.

## S0 serving and route smoke

- Fresh seed `24181`; ten distinct start-position posterior cells excluding prior
  observations of rock 5.
- Temperature zero and one bounded correction response.

All requirements must pass:

1. Ten cells complete with four dynamically legal fixed-root plans.
2. Exactly ten accepted logical calls, zero forced exits, and zero scoring-time model
   calls; all reasoning usage is retained.
3. `move-SOUTH, move-SOUTH, check-5` appears in at least `8/10` cells.

Any failure stops this model-selection line without another seed, prompt, reasoning
budget, or threshold change.

## Conditional S1 proposal-quality gate

S1 runs only after a complete S0 pass.

- Fresh seed `24182`; sixteen distinct strict h3-over-h2 opportunities.
- Matched random tails use the identical machine-fixed roots.
- Shared-plan d2, strong exhaustive d2 root, exhaustive open-loop h3, and 5,000 paired
  bootstrap replicates are unchanged.

All requirements must pass:

1. Positive 95% lower bound for exact plan value versus matched random tails.
2. Positive 95% lower bound for LLM h3 versus shared-plan d2.
3. Positive 95% lower bound for LLM h3 versus the strong d2 root.
4. At least 75% exhaustive h3 route-root selection.
5. At least 60% mean recovery of the h3-over-strong-d2 opportunity.
6. Complete mechanics and independent replay, zero forced exits, retained reasoning
   accounting, and no scoring-time model calls.

A pass authorizes only a separately frozen paired trajectory confirmation. Expected
S0 plus S1 cost is below `$0.50`, within the `$1` run cap. Registered spend after the
endpoint-free serving calibration is `$38.51879448 / $110`, leaving `$71.48120552`.
