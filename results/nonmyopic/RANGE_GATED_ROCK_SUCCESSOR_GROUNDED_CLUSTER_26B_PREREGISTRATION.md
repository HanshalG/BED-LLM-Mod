# Range-Gated Rock Successor-Grounded Cluster 26B Preregistration

Status: frozen after implementation and deterministic tests, before any direct-vLLM
response under this backend.

## Rationale

The exact Range-Gated Rock benchmark has a registered `+0.46296` entropy-AUC
horizon-three advantage. Non-thinking Mini produced legal tails but found the delayed
route in only `1/10` cells. OpenRouter Gemma 26B and Qwen 14B then accepted zero cells:
their single-call adapters exhausted the total output budget and returned no final
channel. Those lines are closed and supplied no plan-quality evidence.

This fresh backend line uses the repo's previously validated direct-vLLM two-stage
thinking finalizer on the `msc,llm` Slurm partitions, excluding `oat12`:

1. Gemma 4 26B A4B receives up to 4,096 first-pass reasoning tokens.
2. If no final answer is present, code appends the model-specific thinking closer and
   grants a separate 512-token final JSON pass.
3. Forced-finalization events and full reasoning traces are logged. They are not
   scientific failures when the bounded second stage returns a complete legal policy;
   empty or invalid finals still fail closed.
4. The successor-grounded prompt and K4 exact verifier are unchanged: successor and
   rock coordinates are exposed, but no EIG, plan value, rank, preferred tail,
   selected plan, or truth.

## S0 serving and route smoke

- Fresh seed `24183`; ten distinct start-position posterior cells.
- Temperature zero, one bounded correction response, 4,096 reasoning plus 512 final
  tokens per physical request.

All requirements must pass:

1. Ten cells complete with four dynamically legal fixed-root plans.
2. Exactly ten accepted logical cells; every forced finalization, correction, token,
   and raw response is retained; no scoring-time model calls.
3. `move-SOUTH, move-SOUTH, check-5` appears in at least `8/10` cells.
4. No empty forced final, traceback, OOM, or unaccounted request.

Any failure stops this backend line without a seed, prompt, or budget change.

## Conditional S1 proposal-quality gate

S1 runs only after a complete S0 pass.

- Fresh seed `24184`; sixteen distinct strict h3-over-h2 opportunities.
- Identical-root matched-random tails, shared-plan d2, strong exhaustive d2 root,
  exhaustive open-loop h3, and 5,000 paired bootstraps.

All requirements must pass:

1. Positive 95% lower bounds versus matched random, shared-plan d2, and strong d2.
2. At least 75% exhaustive h3 route-root selection.
3. At least 60% mean recovery of the h3-over-strong-d2 opportunity.
4. Complete mechanics, retained serving accounting, no scoring-time model calls, and
   independent replay of roots, plans, values, controls, aggregates, and gates.

A pass authorizes only a separately frozen paired trajectory confirmation. This line
uses local cluster inference and adds no OpenRouter spend.
