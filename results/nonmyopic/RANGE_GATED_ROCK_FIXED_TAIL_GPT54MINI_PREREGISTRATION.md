# Range-Gated Rock Fixed-Tail GPT-5.4 Mini Preregistration

Status: frozen after implementation and deterministic tests, before any model response
under this interface.

## Motivation and interface

The independently audited Range-Gated RockSample[7,8] qualification has a strict
horizon-three opportunity: remote checks are `0.55` accurate, on-site checks are
`0.95` accurate, and rock 5 is two moves south of the start. Over 500 paired trials,
exact receding-horizon d3 beat exact d2 by `+0.46296` entropy-AUC nats with 95% CI
`[+0.45998,+0.46573]`; d3 always moved twice to inspect on site while d2 checked
remotely.

The first 26B named-plan interface failed before scoring because the model violated a
set-level requirement for two distinct movement roots and two distinct check roots.
That line remains stopped. This fresh interface removes only that serving failure:

1. Code fixes four distinct roots: the two legal movements whose successor positions
   are nearest a rock, then the two checks with highest exact one-step EIG.
2. For each fixed root, the model returns exactly two action strings completing a
   legal three-action plan.
3. The prompt exposes the grid, coordinates, rock marginals, sensor law, history, and
   root slots, but no EIG, plan value, preferred tail, selected plan, or truth.
4. Exact code scores all four plans and executes only the selected first action.
   Planning is repeated after the real transition; scoring and belief updates make no
   model calls.

At the registered start position the first fixed movement root is `move-SOUTH`, but
the required second move and on-site check are not supplied.

## S0 serving and route smoke

- Model: `openai/gpt-5.4-mini` through OpenRouter, non-thinking, temperature zero.
- Fresh seed `24177`; ten distinct start-position posterior cells with zero to two
  weak remote observations that exclude rock 5.
- K4 fixed roots, horizon three, 256-token output cap, one bounded correction response.

All requirements must pass:

1. Ten cells complete with four dynamically legal plans and the exact fixed roots.
2. Exactly ten accepted logical calls; zero reasoning tokens, forced exits, and
   scoring-time model calls.
3. The delayed on-site plan
   `move-SOUTH, move-SOUTH, check-5` is present in at least `8/10` cells.

Any failure stops this interface without another seed or prompt repair.

## Conditional S1 proposal-quality gate

S1 runs only after a complete S0 pass.

- Fresh seed `24178`; sixteen distinct start-position posterior cells.
- Every cell must independently retain the strict exact h3-over-h2 opportunity.
- LLM K4 plans are scored exactly.
- Matched random samples one legal two-action tail under each of the exact same four
  machine-fixed roots.
- Shared-plan d2 truncates the same LLM plans to two actions, selects under d2, and
  evaluates the corresponding full plan under the common h3 score.
- Strong d2 chooses the exhaustive adaptive d2 root, then receives the best exhaustive
  open-loop h3 continuation under that root.
- Exhaustive open-loop h3 supplies the opportunity ceiling.
- 5,000 paired bootstrap replicates.

All requirements must pass:

1. LLM-minus-matched-random plan value has a strictly positive 95% lower bound.
2. LLM-h3-minus-shared-plan-d2 has a strictly positive 95% lower bound.
3. LLM-h3-minus-strong-d2-root has a strictly positive 95% lower bound.
4. The exact verifier selects the exhaustive h3 route root in at least 75% of cells.
5. Mean recovery of the exact h3-over-strong-d2 opportunity is at least 60%.
6. All cells, plans, controls, and usage checks pass, with zero reasoning, forced
   exits, and scoring-time model calls.
7. Independent replay reproduces roots, compiled plans, values, controls, aggregates,
   and every gate.

A pass authorizes only a separately frozen paired trajectory confirmation. No S0 or
S1 cell may enter that endpoint. Expected S0 plus S1 cost is below `$0.20`, within the
existing `$6` run cap. Registered project spend is `$38.50724716 / $110`, leaving
`$71.49275284`.
