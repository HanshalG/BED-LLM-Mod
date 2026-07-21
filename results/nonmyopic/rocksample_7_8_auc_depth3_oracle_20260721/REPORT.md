# RockSample[7,8] AUC-Aligned Exact Depth Qualification

Positive paired gains favor the deeper AUC-aligned policy.

| Comparison | Entropy-AUC gain [95% CI] | Truth-log-AUC gain [95% CI] | W/T/L |
| --- | --- | --- | --- |
| d2 minus d1 | +0.5768 [+0.5697, +0.5838] | +0.5689 [+0.5399, +0.5974] | 500/0/0 |
| d3 minus d2 | +0.2744 [+0.2689, +0.2799] | +0.2797 [+0.2613, +0.2977] | 500/0/0 |
| d3 minus d1 | +0.8513 [+0.8472, +0.8551] | +0.8486 [+0.8268, +0.8707] | 500/0/0 |

- Primary d3-over-d2 gate: **True**.
- Truth-log corroboration: **True**.
- Initial AUC-aligned values: `{'1': 0.009185981503181395, '2': 0.06928311641999674, '3': 0.6931471805599418}`.
- Mechanics: `{'paired_trial_truths': True, 'all_selected_actions_legal': True, 'all_traces_have_registered_rounds': True, 'no_llm_calls': True}`.
