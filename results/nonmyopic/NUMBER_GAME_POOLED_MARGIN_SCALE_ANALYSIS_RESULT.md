# Number Game Pooled-Margin Scale Analysis Result

Run: `number-game-pooled-margin-scale-analysis-20260729T103359Z`

Status: **retrospective scale-calibration null**.

## Result

The frozen dimensionless score divides each changed-root predicted advantage
by the population standard deviation of that tree's eight candidate-root
risks.

- raw margin Spearman: `-0.0367075`;
- normalized margin Spearman: `0.0309232`;
- normalized 95% tree-bootstrap interval:
  `[-0.296275, 0.369925]`;
- normalized-minus-raw Spearman: `0.0676307`;
- paired difference interval: `[-0.165405, 0.316593]`.

Both frozen positivity gates fail. Candidate-risk population SD ranges from
`0.0081304` to `0.0193433`, with mean `0.0127758`; its correlation with
realized advantage is only `-0.0643`.

## Interpretation

Tree-specific raw-risk scale is not the missing calibration mechanism.
Standardization moves the point estimate slightly upward but leaves it near
zero and imprecise. No alternative denominator, clipping, rank transform,
subset, or regularizer is tried.

The source pooled confirmation remains `gated_null`. This diagnostic made
zero model calls and cost `$0`.

Public `RESULT.json` SHA-256:
`7d68fd2f7b03e3636f70cac29a2c5f771a811a45ad217950fd550a6fe01c37da`.
