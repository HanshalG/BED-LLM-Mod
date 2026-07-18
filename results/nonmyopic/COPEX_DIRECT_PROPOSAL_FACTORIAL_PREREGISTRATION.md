# COPEx Direct-Proposal Depth x Proposal Factorial Preregistration

Frozen on 2026-07-18 before any live result from this runner is read.

## Question

Does exact depth-two sequential EIG improve over one-step EIG when the LLM supplies
small legal continuous **action proposal pools**, rather than previously unsuccessful
natural-language multi-step plans?  Does that depth gain depend on LLM proposal
quality relative to a fixed angular grid?

## Task and Scoring

The task is the repository's continuous COPEx `Location_budgeted` implementation:
one source is uniform on `[0,1]^2`; an action at `x` observes
`log(0.1 + (1e-4 + ||x-theta||^2)^-1)` plus Gaussian noise with sd `0.5`; every
successive action must have L-infinity displacement at most `0.1`.  The finite
particle posterior includes the realized source exactly.  Signal evaluations,
Gaussian likelihood updates, counterfactual observations, entropy, and posterior
mean decoding are programmatic.

The LLM emits only JSON cells of exactly three continuous direction angles. The
executor deterministically converts each angle into the maximum legal L-infinity step,
clipping only at the box boundary. At a boundary, distinct angles can map to the same
endpoint or a no-op; such endpoints are deduplicated without replacement and never
padded. The LLM does not receive outcome likelihoods, score candidates, execute
actions, update beliefs, or write strategies. In d2, the same interface is used at
simulated one-observation child belief states; these are counted as inner proposal
calls and saved verbatim.

Depth-two root values use four common-random-number outer source/noise draws. Each
branch's child actions are ranked with eight CRN immediate-EIG draws. These are Monte
Carlo estimates of an otherwise exact finite-support model, never LLM-generated
scores. All candidates inside a decision receive the same relevant random draws.

## Pilot Configuration

- Fresh seed `46021`; 8 paired trials; 8 sequential queries each.
- 48 uniform prior particles plus the true source; uniformly sampled shared initial
  sensor position and shared realized observation innovations across arms.
- Generator: `google/gemma-4-26b-a4b-it`, OpenRouter, `thinking: false`,
  temperature `0`, output cap 512, adapter concurrency cap 128.
- Candidate width 3; depth-two outer rollouts 4; child immediate rollouts 8.
- One validation-feedback retry; a cell that still fails ends the run closed rather
  than being repaired or padded programmatically.
- The pilot has a `$2.50` run cap. It is feasibility/direction only and will not be
  promoted to a result without a freshly seeded confirmation.

## Arms and Controls

1. `llm_d1`: immediate exact EIG over the root LLM move cell.
2. `llm_d2`: depth-two EIG over that same root cell and LLM cells generated at each
   sampled child belief state.
3. `llm_width`: immediate EIG over one root cell plus additional current-state LLM
   cells. It receives exactly the `1 + K_realized * outer_rollouts` proposal calls
   allocated virtually by `llm_d2` at each nonterminal decision, where `K_realized`
   is the number of distinct physical root endpoints after boundary projection.
4. `grid_d1`: immediate EIG over three fixed angular-grid legal actions.
5. `grid_d2`: depth-two EIG using the same three-action angular-grid interface at
   roots and child states.

When the d1 and d2 LLM arms begin in the same state, their root cell is cached and
identical.  Calls after their selected histories diverge are intentionally not shared.

## Endpoints and Promotion

Primary endpoint: paired **entropy-AUC reduction** over all eight selected queries:
`mean_t H_baseline(t) - mean_t H_llm_d2(t)`. Positive values favor `llm_d2`.
Report a 10,000-replicate paired bootstrap 95% interval and wins/ties/losses against
`llm_d1`, `llm_width`, `grid_d1` through the d2-depth contrast, and `grid_d2`.

Secondary endpoints: final entropy, true-particle log-posterior AUC and final value,
and RMSE. The depth-by-proposal interaction is
`(llm_d2 - llm_d1) - (grid_d2 - grid_d1)` on entropy AUC.

The pilot is promotable only if the mean entropy-AUC gain of `llm_d2` is positive
against both `llm_d1` and `llm_width`, all selected moves are legal, root cells are
shared at the initial paired state, and width's logical proposal allocation equals
d2's virtual allocation. A promotable pilot will be rerun at 30 fresh paired trials
with no parameter changes other than seed and trial count.
