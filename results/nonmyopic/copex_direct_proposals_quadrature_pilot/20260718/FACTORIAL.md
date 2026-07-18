# COPEx Direct-Proposal Depth x Proposal Factorial

The LLM proposes only legal next sensor locations. Likelihoods, finite-support posterior updates, counterfactual observations, and all scoring are programmatic.

| Arm | Entropy AUC | Final entropy | Truth-log-posterior AUC | Final RMSE | Mean LLM proposal cells / decision |
| --- | ---: | ---: | ---: | ---: | ---: |
| llm_d1 | 1.9505 | 1.0665 | -1.8497 | 0.0732 | 1.0 |
| llm_d2 | 1.9879 | 1.2031 | -1.9190 | 0.0816 | 21.6 |
| llm_width | 1.9008 | 0.8032 | -1.8505 | 0.0636 | 21.6 |
| grid_d1 | 2.3161 | 1.6067 | -2.1411 | 0.1014 | 0.0 |
| grid_d2 | 2.1688 | 1.4303 | -1.9778 | 0.1273 | 0.0 |

| Comparison (positive favors LLM d2 / depth) | Entropy-AUC gain | 95% paired bootstrap CI | W / T / L |
| --- | ---: | --- | --- |
| llm_d2_minus_llm_d1 | -0.0374 | [-0.3496, +0.2373] | 5 / 1 / 2 |
| llm_d2_minus_llm_width | -0.0870 | [-0.3779, +0.2280] | 3 / 1 / 4 |
| grid_d2_minus_grid_d1 | +0.1474 | [+0.0095, +0.3305] | 6 / 1 / 1 |
| llm_d2_minus_grid_d2 | +0.1809 | [-0.2646, +0.6138] | 6 / 0 / 2 |
| llm_d2_minus_llm_d1_truth_log_probability | -0.0693 | [-0.4459, +0.2136] | 5 / 1 / 2 |

Depth-by-proposal interaction: `-0.1847` nats entropy-AUC, 95% CI [-0.6265, +0.1342].

## Mechanics

- terminal_cell_failures: `0`.
- all_actions_legal: `True`.
- initial_root_cell_shared: `True`.
- width_call_allocation_matches_virtual_depth_two: `True`.
- physical_llm_requests: `2777`.
- accepted_llm_cells: `2777`.
- raw_rejected_responses: `0`.
- logical_llm_calls: `2832`.
- cache_hits: `55`.
- inner_llm_calls_used_only_for_action_proposals: `True`.
