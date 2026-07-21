# RockSample[7,8] AUC-Aligned Depth Result

AUC-aligned exact planning restores monotonic depth under the same utility. Positive paired gains favor the deeper arm.

| Comparison | Entropy-AUC gain [95% CI] | Truth-log-AUC gain [95% CI] | W/T/L |
| --- | --- | --- | --- |
| d2 minus d1 | +0.5768 [+0.5697, +0.5838] | +0.5689 [+0.5399, +0.5974] | 500/0/0 |
| d3 minus d2 | +0.2744 [+0.2689, +0.2799] | +0.2797 [+0.2613, +0.2977] | 500/0/0 |
| d3 minus d1 | +0.8513 [+0.8472, +0.8551] | +0.8486 [+0.8268, +0.8707] | 500/0/0 |

The repair is real but bounded: aligned d3 exactly matches the previous terminal-EIG d2 action sequence, round-entropy curve, entropy AUC (`4.6440861`), and final entropy (`3.4657359`). It fixes the d3 receding-horizon failure but creates no strict oracle gain over the strongest d2 policy.

- Same-utility monotonicity gate: **True**.
- Truth-log corroboration: **True**.
- Strict gain over prior best d2: **False**.
- Independent trace audit: **True**.
