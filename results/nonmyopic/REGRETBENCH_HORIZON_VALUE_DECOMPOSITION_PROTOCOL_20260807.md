# RegretBench Horizon-Value Decomposition Protocol

Date frozen: 2026-08-07, before any RegretBench policy response or endpoint.

## Purpose

Distinguish two causes of a non-myopic null:

1. the simulator forecasts little action-dependent delayed value, leaving no
   useful horizon signal to exploit; or
2. it forecasts material delayed value, but that value does not predict the
   realized endpoint.

This is a zero-call, post-result, descriptive diagnostic. It cannot alter a
root, gate, result status, authorization, confirmation decision, claim tier,
model, draw count, prompt, subset, or endpoint.

## Cohort

Run on every complete independently verified original or SMC RegretBench
development or confirmation result. Use only tasks where dynamic depth two and
refresh-matched myopic select different first roots. Do not report a favorable
subset beyond this prospectively required changed-root set.

If mechanics failed, report `unavailable_mechanics_failed`. Fewer than 16
changed-root tasks are `insufficient_changed_roots`.

## Exact Per-Task Decomposition

Let `d` be the dynamic-selected root and `m` the refresh-myopic root. Let
`I(r)` be `myopic_refresh_brier_root_risks[r].brier`, `T(r)` be
`conditioned_root_risks[r].brier`, and `R(policy)` be realized aligned Brier.

Compute:

```text
immediate_penalty = I(d) - I(m)
dynamic_horizon_value = I(d) - T(d)
refresh_horizon_value = I(m) - T(m)
differential_horizon_value = dynamic_horizon_value - refresh_horizon_value
predicted_terminal_advantage = T(m) - T(d)
realized_terminal_advantage = R(m) - R(d)
```

Require the exact identity, within `1e-10`:

```text
predicted_terminal_advantage
= differential_horizon_value - immediate_penalty
```

The dynamic and refresh selectors imply nonnegative predicted terminal
advantage and immediate penalty up to numerical tolerance. A violation means
the artifact is inconsistent and the diagnostic must fail.

## Metrics

Across changed-root tasks report means, sample standard deviations, and
task-bootstrap 95% intervals plus probability positive for all six quantities.
Also report:

- Spearman correlation, bootstrap interval, and probability positive between
  predicted and realized terminal advantage;
- sign accuracy after excluding exact-zero pairs;
- RMSE and MAE between predicted and realized terminal advantage; and
- dynamic wins, ties, and losses against refresh-myopic on the realized paired
  roots.

Use 20,000 task bootstraps with seed `202608430000`. No alternative seed,
threshold, transform, or winsorization may be selected after looking.

## Ordered Interpretation

Apply the first matching region:

1. `unavailable_mechanics_failed`.
2. `insufficient_changed_roots`: fewer than 16 changed roots or fewer than 90%
   valid bootstrap correlations.
3. `no_material_horizon_forecast`: mean differential horizon value below
   `.01` or bootstrap probability positive below `.80`.
4. `forecast_horizon_not_realized`: the preceding forecast gate passes, but
   mean realized advantage is nonpositive, sign accuracy is below `.55`, or
   predicted-realized Spearman is below `.15` / probability positive below
   `.80`.
5. `descriptive_horizon_value_supported`: forecast gate passes, mean realized
   advantage is at least `.02`, its 95% lower bound is positive, sign accuracy
   is at least `.60`, and Spearman is at least `.15` with probability positive
   at least `.80`.
6. `partial_horizon_value_evidence`: all other valid cases.

These regions diagnose mechanism only. Even
`descriptive_horizon_value_supported` cannot rescue a preregistered null or
authorize confirmation.
