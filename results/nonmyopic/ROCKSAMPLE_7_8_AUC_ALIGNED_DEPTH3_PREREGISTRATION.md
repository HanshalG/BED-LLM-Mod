# RockSample[7,8] AUC-Aligned Depth-Three Qualification

Registered 2026-07-21 after auditing the terminal-EIG d3 failure and before running
any AUC-aligned trajectory endpoint.

## Hypothesis

The failed exact d3 policy maximized information at the end of each rolling
three-step window, while the registered policy endpoint averages entropy across all
rounds. At the first divergence it selected a noisy check because its terminal value
included a promised later move and perfect check; replanning repeatedly deferred that
continuation.

For an `h`-step window, minimizing expected entropy AUC is equivalent to weighting the
first incremental EIG by `h`, the next by `h-1`, and the final gain by one. The frozen
repair changes only those exact Bellman weights. Dynamics, prior, likelihood,
observation coupling, legal actions, horizon, and evaluation remain unchanged.

## Frozen Design

- Canonical diagnosis-only RockSample[7,8].
- Fresh seed `24076`.
- `500` paired truths, `10` rounds, exact exhaustive d1/d2/d3.
- Both d2 and d3 use the same AUC-aligned recursive utility; d1 is unchanged.
- Common-random-number observation coupling, `10,000` paired bootstraps, and the same
  entropy-AUC, truth-log-AUC, final entropy, and MAP metrics as the prior exact gate.
- Zero LLM calls, generated proposals, Monte Carlo rollouts, reward, sampling, or exit.

## Gate

Positive gains favor d3. The primary gate passes only if the 95% paired bootstrap
lower bound for AUC-aligned d3 minus AUC-aligned d2 entropy-AUC gain is strictly above
zero. Truth-log-AUC requires the same positive lower bound for corroboration. A tie or
negative interval stops this repair. The prior terminal-EIG d2 curve is an external
reference, not a substitute for the same-utility d2 control.

Only a passed exact gate can authorize a separately preregistered LLM h3 policy-tree
instrument. This exact run does not itself establish proposal quality.

## Frozen Command

```bash
python scripts/nonmyopic_rock_auc_depth_oracle.py \
  --map 7-8 --num-trials 500 --num-rounds 10 --max-depth 3 \
  --seed 24076 --bootstrap-replicates 10000 --trial-concurrency 16 \
  --output-dir results/nonmyopic/rocksample_7_8_auc_depth3_oracle_20260721
```
