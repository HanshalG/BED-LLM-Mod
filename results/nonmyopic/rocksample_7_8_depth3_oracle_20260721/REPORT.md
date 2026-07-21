# RockSample[7,8] Exact Depth-Three Qualification

Positive paired gains favor the deeper exact policy.

| Comparison | Entropy-AUC gain [95% CI] | Truth-log-AUC gain [95% CI] | AUC W/T/L |
| --- | --- | --- | --- |
| d2 minus d1 | +0.8509 [+0.8468, +0.8548] | +0.8499 [+0.8278, +0.8726] | 500/0/0 |
| d3 minus d2 | -0.2044 [-0.2090, -0.1998] | -0.2196 [-0.2396, -0.1999] | 0/0/500 |
| d3 minus d1 | +0.6465 [+0.6404, +0.6526] | +0.6304 [+0.6015, +0.6588] | 500/0/0 |

- Primary d3-over-d2 gate: **False**.
- Truth-log corroboration: **False**.
- Initial exact values: `{'1': 0.009185981503181395, '2': 0.06928311641999674, '3': 0.6931471805599418}`.
- Mechanics: `{'paired_trial_truths': True, 'all_selected_actions_legal': True, 'all_traces_have_registered_rounds': True, 'no_llm_calls': True}`.
