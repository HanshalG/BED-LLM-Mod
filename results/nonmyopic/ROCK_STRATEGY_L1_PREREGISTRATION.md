# Rock Strategy-Prior L1 Preregistration

Frozen before any live LLM strategy result is generated or read. Date: 2026-07-16.

## Objective

Test whether an LLM prior over compact contingent strategies is load-bearing in exact
non-myopic BED, rather than merely supplying more proposals or scoring compute.

## Interface-Only Cost Smoke

Before the formal run, one bounded live interface smoke may run with seed `12031`, one
trajectory per map, two rounds, K=4, and horizon 2. Only completion status, raw
parse/validation failures, root-mix validity, request/token counts, and cost may be
read. Policy metrics are quarantined and must not influence the formal configuration.
The prior deterministic dry smoke at seed `12031` is also mechanics-only and excluded
from evidence.

The live smoke passes if every cell either validates immediately or after the single
registered feedback retry; all actions are legal; no terminal cell fails; width call
and exact-scorer units match StrategyEIG; and cost supports the frozen formal cap. An
interface-only defect may be repaired before the formal launch without reading policy
outcomes. No reward, exact scorer, arm, endpoint, or pass-rule change is permitted.

## Formal Configuration

- Maps: Rock Diagnosis Figure 4 `3-6` and `5-7`.
- Fresh formal seed: `12032`.
- 30 paired trajectories per map, 8 rounds per trajectory.
- Strategy count K=4; planning horizon 2; receding-horizon regeneration.
- Generator: `google/gemma-4-26b-a4b-it`, non-thinking, temperature 0.
- Strategy grammar: the strict compact reactive grammar frozen in commit `774c67d`,
  with at least one move root and one check root in every horizon-2 cell.
- One validation-feedback retry maximum; then fail closed. No action padding,
  substitution, or parser fallback.
- Exact full-vector posterior, observation likelihood, rollout EIG, outcome sampling,
  and MAP decode. Strategy scoring makes zero LLM calls.
- Paired truths and keyed observation noise across arms.

## Arms

1. `strategy_eig`: K LLM strategies, exact horizon-2 rollout EIG, execute the winning
   root.
2. `exhaustive_d2`: exact depth-2 optimization over every legal action, the short-
   horizon gold anchor.
3. `shared_d1`: the same LLM strategy cell as StrategyEIG whenever states coincide,
   scored only by exact immediate EIG of its root actions.
4. `width`: one LLM call returns a full legal-action ordering; one-step exact EIG uses
   exactly StrategyEIG's expanded decision-node budget. Once all distinct legal actions
   are exhausted, remaining units are repeated and logged.
5. `random_strategy`: K strategies sampled from the same grammar and conditioned on
   the same move/check root-mix invariant, then scored by the same exact horizon-2 EIG.

## Endpoints and Frozen Analysis

Primary endpoint: paired final posterior entropy gain in nats,
`H_baseline(T) - H_strategy_eig(T)`, computed separately on each map against
`shared_d1`, `width`, and `random_strategy`.

For each of the six map-by-baseline comparisons, report the mean paired gain, a 10,000-
replicate paired bootstrap 95% interval, and wins/ties/losses. The L1 gate passes only
if the lower endpoint is strictly above zero for **all six comparisons**. This is an
intersection-union gate; no single favorable comparison can substitute for another.

Secondary diagnostics, never substitutes for the primary gate:

- final truth log posterior probability and MAP accuracy;
- entropy, truth-log, and MAP traces;
- exact StrategyEIG value as a fraction of exhaustive d2 value;
- exhaustive node counts and width saturation;
- verbatim candidate and winning strategies;
- raw invalid responses, retry count, logical/physical calls, token use, and cost.

## Cost and Decision Rule

Pre-smoke conservative formal projection: `$0.80`; hard per-run cap: `$1.00`; project
budget remains `$40`. Replace the projection with the measured interface-smoke
extrapolation before launch if it is larger. Do not launch if the projected formal cost
exceeds the hard cap or remaining authorization.

Pass or partial (beats controls but captures little exhaustive value) proceeds to L2.
Failure versus random triggers exactly one prompt/grammar quality diagnosis and one
interface-only rerun under the goal's branch rule. A second random-control failure
pivots to L3 without a third Rock attempt.
