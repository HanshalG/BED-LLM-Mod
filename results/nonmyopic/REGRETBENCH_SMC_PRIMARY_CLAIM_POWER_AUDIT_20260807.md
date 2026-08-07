# RegretBench SMC Primary-Claim Power Audit

This zero-call diagnostic sends synthetic paired task rows through the exact frozen 13-gate scorer. It is an operating-characteristic audit, not a forecast of live DeepSeek behavior.

| Scenario | Refresh changes | Blind changes | Refresh gain | Blind gain | Target rho | One-cohort pass | Nominal two-cohort pass | 95% Wilson interval |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| threshold_surface | 16 | 12 | 0.020 | 0.015 | 0.15 | 0.081 | 0.007 | [0.066, 0.100] |
| design_target | 28 | 22 | 0.035 | 0.025 | 0.35 | 0.766 | 0.587 | [0.739, 0.791] |
| strong_signal | 36 | 28 | 0.050 | 0.035 | 0.50 | 0.977 | 0.955 | [0.966, 0.985] |

## Gate Pass Rates

### threshold_surface

- `dynamic_refresh_myopic_differ_at_least_16`: `1.000`
- `predicted_gain_over_refresh_myopic_at_least_001`: `0.691`
- `dynamic_refresh_myopic_brier_gain_at_least_002`: `0.530`
- `dynamic_refresh_myopic_probability_at_least_090`: `0.995`
- `dynamic_refresh_myopic_wins_exceed_losses`: `1.000`
- `dynamic_log_loss_nonworse_refresh_myopic`: `1.000`
- `refresh_myopic_predicted_realized_spearman_at_least_015`: `0.503`
- `refresh_myopic_spearman_probability_positive_at_least_080`: `0.374`
- `dynamic_blind_differ_at_least_12`: `1.000`
- `dynamic_blind_brier_gain_at_least_0015`: `0.514`
- `dynamic_blind_probability_at_least_080`: `0.993`
- `dynamic_blind_wins_exceed_losses`: `0.995`
- `dynamic_log_loss_nonworse_blind`: `1.000`

### design_target

- `dynamic_refresh_myopic_differ_at_least_16`: `1.000`
- `predicted_gain_over_refresh_myopic_at_least_001`: `1.000`
- `dynamic_refresh_myopic_brier_gain_at_least_002`: `0.994`
- `dynamic_refresh_myopic_probability_at_least_090`: `1.000`
- `dynamic_refresh_myopic_wins_exceed_losses`: `1.000`
- `dynamic_log_loss_nonworse_refresh_myopic`: `1.000`
- `refresh_myopic_predicted_realized_spearman_at_least_015`: `0.836`
- `refresh_myopic_spearman_probability_positive_at_least_080`: `0.805`
- `dynamic_blind_differ_at_least_12`: `1.000`
- `dynamic_blind_brier_gain_at_least_0015`: `0.961`
- `dynamic_blind_probability_at_least_080`: `1.000`
- `dynamic_blind_wins_exceed_losses`: `1.000`
- `dynamic_log_loss_nonworse_blind`: `1.000`

### strong_signal

- `dynamic_refresh_myopic_differ_at_least_16`: `1.000`
- `predicted_gain_over_refresh_myopic_at_least_001`: `1.000`
- `dynamic_refresh_myopic_brier_gain_at_least_002`: `1.000`
- `dynamic_refresh_myopic_probability_at_least_090`: `1.000`
- `dynamic_refresh_myopic_wins_exceed_losses`: `1.000`
- `dynamic_log_loss_nonworse_refresh_myopic`: `1.000`
- `refresh_myopic_predicted_realized_spearman_at_least_015`: `0.979`
- `refresh_myopic_spearman_probability_positive_at_least_080`: `0.979`
- `dynamic_blind_differ_at_least_12`: `1.000`
- `dynamic_blind_brier_gain_at_least_0015`: `0.999`
- `dynamic_blind_probability_at_least_080`: `1.000`
- `dynamic_blind_wins_exceed_losses`: `1.000`
- `dynamic_log_loss_nonworse_blind`: `1.000`

## Interpretation

The threshold-surface scenario is expected to have low conjunction power because several observed statistics sit exactly on one-sided decision boundaries. Under the design-target scenario, paired effect gates are well powered and changed-root ranking fidelity is the dominant failure mode. A strong signal is detected reliably. The audit therefore supports retaining 64 tasks while treating root diversity and predicted-to-realized fidelity as the key live diagnostics.

Secondary controls and log-loss safeguards are held nonbinding here, so these rates must not be presented as unconditional probabilities of a live pass.
The two-cohort column is only the square of the one-cohort Monte Carlo rate under an independence assumption. It is not a forecast because model-level errors can be shared across cohorts.

A 100-replicate implementation calibration using these same three scenarios was inspected before the 1,000-replicate artifact; no scenario, parameter, scorer, or gate changed afterward.
