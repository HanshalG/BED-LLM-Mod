# RegretBench SMC Primary Claim-Gate Amendment

Date frozen: 2026-08-07, before any RegretBench SMC response.

Status: prospective classification correction. It changes no model, prompt,
particle, likelihood, policy selection, task, seed, request, endpoint, metric,
threshold, control, bootstrap, or budget.

## Problem

The shared pre-SMC scorer emits 34 scientific booleans. The original SMC
protocol made their full conjunction determine `passed`. That allows
fixed-parent EIG, fixed-parent Brier, fixed-depth-two, and their separate
calibration checks to veto the stated SMC claim even when the transition-
matched myopic and history-blind comparisons pass. Those controls test useful
but different estimands. Treating every reported diagnostic as a co-primary
gate is not a stronger test of the headline claim; it is a multiple-endpoint
false-negative mechanism.

This issue was found from source inspection before any SMC response. No result,
task-level endpoint, or model output informed this amendment.

## Frozen Primary Conjunction

An SMC development or confirmation result is `passed` only if mechanics pass
and all 13 booleans below are true.

Headline horizon-isolating comparison against
`smc_myopic_refresh_brier`:

1. `dynamic_refresh_myopic_differ_at_least_16`;
2. `predicted_gain_over_refresh_myopic_at_least_001`;
3. `dynamic_refresh_myopic_brier_gain_at_least_002`;
4. `dynamic_refresh_myopic_probability_at_least_090`;
5. `dynamic_refresh_myopic_wins_exceed_losses`;
6. `dynamic_log_loss_nonworse_refresh_myopic`;
7. `refresh_myopic_predicted_realized_spearman_at_least_015`; and
8. `refresh_myopic_spearman_probability_positive_at_least_080`.

Path-dependence comparison against `smc_history_blind_depth2`:

9. `dynamic_blind_differ_at_least_12`;
10. `dynamic_blind_brier_gain_at_least_0015`;
11. `dynamic_blind_probability_at_least_080`;
12. `dynamic_blind_wins_exceed_losses`; and
13. `dynamic_log_loss_nonworse_blind`.

The first family asks whether valuing the second question improves the exact
same generated-transition utility relative to one-step choice. The second
asks whether visible-answer-conditioned LLM particle dynamics add value over
matched history-blind transitions. Both are required because the paper claim
is specifically non-myopic planning over path-dependent LLM belief dynamics.

## Mandatory Non-Gating Diagnostics

All 34 original booleans remain serialized under their existing names. The
result also reports:

- `primary_claim_gate_names` and each corresponding value;
- `primary_claim_all_pass`;
- `all_34_diagnostic_gates_pass`; and
- all paired controls, effects, intervals, probabilities, wins/ties/losses,
  disagreements, correlations, and draw-stability diagnostics.

Fixed-parent Brier, entropy EIG, fixed-depth-two, random, fresh-regeneration,
alignment-complete, optional-baseline, subgroup, and pooled analyses cannot
rescue a failed primary conjunction or veto a passed one. They remain required
for interpretation and reviewer-visible failure analysis.

## Status And Claims

- mechanics failure remains `mechanics_failed` with no efficacy result;
- mechanics pass plus any failed primary claim gate is `gated_null`;
- mechanics pass plus all 13 primary claim gates is `passed`;
- development `passed` remains provisional and requires untouched
  confirmation; and
- only an independently verified confirmation `passed` authorizes the
  confirmed paper tier.

The producer and producer-independent verifier must implement the 13-key list
separately and agree exactly. Freezing this amendment makes zero model calls
and costs zero dollars.
