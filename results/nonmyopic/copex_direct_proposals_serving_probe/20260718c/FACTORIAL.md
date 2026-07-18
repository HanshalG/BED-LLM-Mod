# COPEx Direct-Proposal Depth x Proposal Factorial

The LLM proposes only legal next sensor locations. Likelihoods, finite-support posterior updates, counterfactual observations, and all scoring are programmatic.

| Arm | Entropy AUC | Final entropy | Truth-log-posterior AUC | Final RMSE | Mean LLM proposal cells / decision |
| --- | ---: | ---: | ---: | ---: | ---: |
| llm_d1 | 2.5675 | 2.1594 | -2.6663 | 0.1650 | 1.0 |
| llm_d2 | 2.5675 | 2.1594 | -2.6663 | 0.1650 | 7.0 |
| llm_width | 2.5675 | 2.1594 | -2.6663 | 0.1650 | 7.0 |
| grid_d1 | 2.1800 | 1.9807 | -3.0962 | 0.2761 | 0.0 |
| grid_d2 | 2.1800 | 1.9807 | -3.0962 | 0.2761 | 0.0 |

| Comparison (positive favors LLM d2 / depth) | Entropy-AUC gain | 95% paired bootstrap CI | W / T / L |
| --- | ---: | --- | --- |
| llm_d2_minus_llm_d1 | +0.0000 | [+0.0000, +0.0000] | 0 / 1 / 0 |
| llm_d2_minus_llm_width | +0.0000 | [+0.0000, +0.0000] | 0 / 1 / 0 |
| grid_d2_minus_grid_d1 | +0.0000 | [+0.0000, +0.0000] | 0 / 1 / 0 |
| llm_d2_minus_grid_d2 | -0.3875 | [-0.3875, -0.3875] | 0 / 0 / 1 |
| llm_d2_minus_llm_d1_truth_log_probability | +0.0000 | [+0.0000, +0.0000] | 0 / 1 / 0 |

Depth-by-proposal interaction: `+0.0000` nats entropy-AUC, 95% CI [+0.0000, +0.0000].

## Mechanics

- terminal_cell_failures: `0`.
- all_actions_legal: `True`.
- initial_root_cell_shared: `True`.
- width_call_allocation_matches_virtual_depth_two: `True`.
- physical_llm_requests: `26`.
- accepted_llm_cells: `26`.
- raw_rejected_responses: `0`.
- logical_llm_calls: `30`.
- cache_hits: `4`.
- inner_llm_calls_used_only_for_action_proposals: `True`.
