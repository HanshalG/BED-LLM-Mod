# RMSE Repair Analysis

- Records: 60
- Depths: 2, 3, 5

## Realized-Realized Link

| Depth | entropy-drop vs RMSE-drop Spearman | truth-log-prob vs RMSE-drop Spearman | RMSE SNR | query distance ratio |
|---:|---:|---:|---:|---:|
| 2 | 0.101 +/- 0.058 | 0.083 +/- 0.056 | 0.572 +/- 0.164 | 4.782 +/- 0.334 |
| 3 | 0.073 +/- 0.055 | 0.068 +/- 0.051 | 0.424 +/- 0.051 | 2.798 +/- 0.147 |
| 5 | 0.103 +/- 0.049 | 0.098 +/- 0.050 | 0.474 +/- 0.050 | 2.171 +/- 0.104 |

Interpretation: this table tests whether realized posterior-information gains are themselves rank-aligned with realized point-RMSE gains across candidate strategies. If these values are near zero, then point-RMSE is weakly rankable at the probe horizons even when the information metric is rankable.

## Expected Posterior RMSE

Status: `unavailable_from_current_records`.

The aggregate ranking-fidelity JSONL stores candidate-level realized entropy drops, point-RMSE drops, truth-log-probability means, and query-distance diagnostics, but it does not store final posterior hypothesis supports/probabilities for each deployment. Expected posterior RMSE cannot be recomputed exactly without those posterior states.
