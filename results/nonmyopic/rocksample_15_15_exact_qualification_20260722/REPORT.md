# RockSample[15,15] Exact Depth-2 Qualification

Positive paired gains favor the deeper exact policy.

| Comparison | Entropy-AUC gain [95% CI] | Truth-log-AUC gain [95% CI] | AUC W/T/L |
| --- | --- | --- | --- |
| d2 minus d1 | +0.5888 [+0.5801, +0.5971] | +0.5930 [+0.5470, +0.6396] | 100/0/0 |

- Primary d2-over-d1 gate: **True**.
- Truth-log corroboration: **True**.
- Initial exact values: `{'1': 0.0057223690223843215, '2': 0.02985110651988665}`.
- Mechanics: `{'paired_trial_truths': True, 'all_selected_actions_legal': True, 'all_traces_have_registered_rounds': True, 'no_llm_calls': True}`.
