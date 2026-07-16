# COPEx-Task Continuous Strategy-Prior L3 Preregistration

Frozen on 2026-07-16 before any live L3 policy endpoint is generated or read.

## Claim and Task

This rung tests whether an LLM prior over compact continuous spatial plans is
load-bearing when action-tree enumeration is unavailable. It independently implements
the published COPEx `Location_budgeted` task equations: one source uniform on
`[0,1]^2`; mean observation
`log(0.1 + (1e-4 + ||xi-theta||^2)^-1)`; additive Gaussian noise sd `0.5`; and
`||xi_t-xi_(t-1)||_infinity <= 0.1`. It does not run or claim comparison against the
unreleased-checkpoint COPEx planner.

The LLM emits only plan JSON. Signal evaluation, transition projection, analytic
likelihoods, particle posterior updates, CRN rollout simulation, EIG estimation,
selection, realized observations, and posterior-mean decode are programmatic. The EIG
expectation uses 64 common-random-number Monte Carlo rollouts per candidate; likelihood
and posterior updates within each rollout are exact for the finite particle support.

## Interface Smoke

One interface-only smoke may use seed `31001`, one trial, three rounds, K=4, horizon 4,
64 particles, and 16 rollouts. Only completion, parser/constraint mechanics,
call/token/cost accounting, and malformed response classes may be read. Policy
endpoints are quarantined. One interface-only clarification is allowed before the
formal run; no plan may be inserted, padded, or repaired programmatically.

## Formal Configuration

- Fresh seed `31002`; 30 paired trials; 30 selected designs per trial.
- 64 prior particles plus the realized truth; exact finite-support posterior.
- K=4 LLM strategies; receding horizon 4; 64 CRN rollouts per strategy.
- Generator `google/gemma-4-26b-a4b-it`, non-thinking, temperature 0.
- Trial concurrency 24; one validation-feedback retry, then fail closed.
- Initial sensor position sampled uniformly and shared by all arms.
- Paired truths, particles, initial positions, and realized Gaussian noise across arms.

## Plan Grammar

Each plan has a natural-language name and rationale plus four strict movement macros:
`vector`, `toward_rank`, `midpoint_ranks`, and `toward_mean`. Posterior-relative macros
are re-executed after every simulated observation, providing compact branch-adaptive
plans. The executor projects every macro into the task's box and L-infinity transition
constraint by definition. Random plans are sampled from this same grammar.

## Arms and Compute Controls

1. `strategy_eig`: K LLM plans scored over horizon 4; execute the winning root.
2. `shared_d1`: the same LLM cell whenever states coincide, scored on roots only.
3. `width`: one LLM call proposes K times horizon distinct continuous vectors; exact
   one-step scoring uses exactly the same simulated action-step units as StrategyEIG.
4. `random_strategy`: K grammar-matched random plans, same horizon and scorer units.
5. `grid_d2`: fixed-budget discretized receding depth-2 sequences at angular
   resolution 8, using exactly StrategyEIG's scorer units each round.

After the primary run, zero-LLM grid-only sensitivity at resolutions 4 and 16 will
reuse the formal seed schedule. The number of evaluated d2 sequences remains fixed,
while the full enumerable tree sizes (16, 64, and 256 sequences for resolutions 4, 8,
and 16) are reported to expose the resolution cost curve.

## Endpoint and Gate

Primary endpoint: paired final posterior entropy gain in nats,
`H_baseline(T)-H_strategy_eig(T)`, separately against shared d1, width, random plans,
and matched-budget grid d2. Report mean, 10,000-replicate paired bootstrap 95% CI, and
wins/ties/losses. L3 passes only if all four lower endpoints are strictly above zero
and all legality/sharing/compute mechanics pass.

Secondary diagnostics: final and roundwise RMSE/entropy, plan EIG, scorer units,
selected actions, candidate and winning strategies verbatim, invalid responses,
requests/tokens, and cost. The formal hard run cap is `$1.50`; current remaining
project authorization exceeds `$21`.

Pass supports a scoped cross-environment strategy-prior result when consolidated with
the banked Rock depth result. Failure against random means the LLM prior is not shown
load-bearing; the project consolidates that negative result without another
environment or interface rescue.
