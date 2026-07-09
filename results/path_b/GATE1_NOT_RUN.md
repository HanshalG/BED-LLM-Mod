# Gate 1 Not Run

Status: **stopped by the pre-registered Gate 0 rule**.

Gate 0 required the expected-posterior-RMSE scorer to reach macro-average
within-probe Spearman rho of approximately 0.3 against realized posterior-RMSE
reduction at some tested horizon. The canonical depth-1 result is rho 0.231
with a 95% bootstrap interval of [0.098, 0.360]. Rollout-count, horizon, and
support-resampling diagnostics did not produce a passing macro result.

Consequently:

- no Gate 1 Slurm jobs were submitted;
- no five-trial arbitration pilot was run;
- headroom, utility-divergence, override-rate, and selection-frequency checks
  are not measured;
- the arbitration-vs-naive and arbitration-vs-greedy claims are not made;
- Phase 2 was not pre-registered or launched.

This is a successful application of the experiment gate, not missing data. The
authoritative evidence is:

- `results/ranking_fidelity/PATH_B_GATE0_TASK_LOSS.md`
- `results/ranking_fidelity/PATH_B_GATE0_DIAGNOSIS.md`
- `results/ranking_fidelity/path_b_gate0_task_loss.json`
