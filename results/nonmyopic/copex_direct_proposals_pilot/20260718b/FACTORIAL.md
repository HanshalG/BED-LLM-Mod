# COPEx Direct-Proposal Depth x Proposal Factorial

The LLM proposes only legal next sensor locations. Likelihoods, finite-support posterior updates, counterfactual observations, and all scoring are programmatic.

| Arm | Entropy AUC | Final entropy | Truth-log-posterior AUC | Final RMSE | Mean LLM proposal cells / decision |
| --- | ---: | ---: | ---: | ---: | ---: |
| llm_d1 | 1.5859 | 1.0504 | -1.5523 | 0.0704 | 1.0 |
| llm_d2 | 1.6261 | 0.8545 | -1.5101 | 0.0830 | 11.2 |
| llm_width | 1.7766 | 0.9985 | -1.6356 | 0.0587 | 11.2 |
| grid_d1 | 1.6656 | 1.0698 | -1.6533 | 0.1058 | 0.0 |
| grid_d2 | 1.6069 | 0.9596 | -1.6170 | 0.1048 | 0.0 |

| Comparison (positive favors LLM d2 / depth) | Entropy-AUC gain | 95% paired bootstrap CI | W / T / L |
| --- | ---: | --- | --- |
| llm_d2_minus_llm_d1 | -0.0402 | [-0.1761, +0.1119] | 3 / 1 / 4 |
| llm_d2_minus_llm_width | +0.1506 | [-0.1153, +0.4401] | 3 / 0 / 5 |
| grid_d2_minus_grid_d1 | +0.0587 | [-0.0683, +0.2504] | 2 / 1 / 5 |
| llm_d2_minus_grid_d2 | -0.0191 | [-0.2970, +0.2262] | 4 / 0 / 4 |
| llm_d2_minus_llm_d1_truth_log_probability | +0.0422 | [-0.1450, +0.2549] | 3 / 1 / 4 |

Depth-by-proposal interaction: `-0.0989` nats entropy-AUC, 95% CI [-0.3072, +0.1001].

## Mechanics

- terminal_cell_failures: `0`.
- all_actions_legal: `True`.
- initial_root_cell_shared: `True`.
- width_call_allocation_matches_virtual_depth_two: `True`.
- physical_llm_requests: `1443`.
- accepted_llm_cells: `1443`.
- raw_rejected_responses: `0`.
- logical_llm_calls: `1500`.
- cache_hits: `57`.
- inner_llm_calls_used_only_for_action_proposals: `True`.
