# Range-Gated RockSample[7,8] Exact Depth-Three Qualification

Positive paired gains favor the deeper exact policy.

| Comparison | Entropy-AUC gain [95% CI] | Truth-log-AUC gain [95% CI] | W/T/L |
| --- | --- | --- | --- |
| d2 minus d1 | +0.000000 [+0.000000, +0.000000] | +0.001906 [-0.001656, +0.005418] | 0/500/0 |
| d3 minus d2 | +0.462960 [+0.459983, +0.465734] | +0.470138 [+0.450752, +0.489263] | 500/0/0 |
| d3 minus d1 | +0.462960 [+0.459988, +0.465692] | +0.472044 [+0.452765, +0.491539] | 500/0/0 |

- Primary d3-over-d2 gate: **True**.
- Truth-log corroboration: **True**.
- Mechanics: `{'paired_trial_truths': True, 'all_selected_actions_legal': True, 'all_traces_have_registered_rounds': True, 'd2_initial_roots_are_checks': True, 'd3_initial_roots_are_moves': True, 'd3_reaches_onsite_inspection': True, 'no_llm_calls': True}`.
