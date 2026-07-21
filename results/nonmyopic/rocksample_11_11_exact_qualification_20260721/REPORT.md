# RockSample[11,11] Exact Depth-2 Qualification

Positive paired gains favor the deeper exact policy.

| Comparison | Entropy-AUC gain [95% CI] | Truth-log-AUC gain [95% CI] | AUC W/T/L |
| --- | --- | --- | --- |
| d2 minus d1 | +1.1042 [+1.1015, +1.1068] | +1.1129 [+1.0885, +1.1377] | 500/0/0 |

- Primary d2-over-d1 gate: **True**.
- Truth-log corroboration: **True**.
- Initial exact values: `{'1': 0.009185981503184948, '2': 0.0692831164199994}`.
- Mechanics: `{'paired_trial_truths': True, 'all_selected_actions_legal': True, 'all_traces_have_registered_rounds': True, 'no_llm_calls': True}`.
