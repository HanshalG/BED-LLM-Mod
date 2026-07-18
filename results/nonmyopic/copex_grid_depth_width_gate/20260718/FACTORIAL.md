# COPEx Direct-Proposal Depth x Proposal Factorial

The LLM proposes only legal next sensor locations. Likelihoods, finite-support posterior updates, counterfactual observations, and all scoring are programmatic.

| Arm | Entropy AUC | Final entropy | Truth-log-posterior AUC | Final RMSE | Mean LLM proposal cells / decision |
| --- | ---: | ---: | ---: | ---: | ---: |
| llm_d1 | 2.2021 | 1.6718 | -2.2557 | 0.1489 | 1.0 |
| llm_d2 | 2.1921 | 1.6628 | -2.2249 | 0.1401 | 22.0 |
| llm_width | 2.2021 | 1.6718 | -2.2557 | 0.1489 | 22.0 |
| grid_d1 | 1.8795 | 1.0406 | -1.9795 | 0.0990 | 0.0 |
| grid_d2 | 1.9444 | 1.0209 | -1.9927 | 0.0951 | 0.0 |
| grid_score_width | 1.5767 | 0.6709 | -1.6874 | 0.0475 | 0.0 |

| Comparison (positive favors LLM d2 / depth) | Entropy-AUC gain | 95% paired bootstrap CI | W / T / L |
| --- | ---: | --- | --- |
| llm_d2_minus_llm_d1 | +0.0100 | [-0.0094, +0.0295] | 33 / 38 / 29 |
| llm_d2_minus_llm_width | +0.0100 | [-0.0096, +0.0297] | 33 / 38 / 29 |
| grid_d2_minus_grid_d1 | -0.0648 | [-0.1429, +0.0138] | 32 / 26 / 42 |
| grid_d2_minus_grid_score_width | -0.3676 | [-0.5005, -0.2378] | 24 / 0 / 76 |
| llm_d2_minus_grid_d2 | -0.2477 | [-0.3619, -0.1320] | 29 / 0 / 71 |
| llm_d2_minus_llm_d1_truth_log_probability | +0.0308 | [-0.0055, +0.0726] | 34 / 38 / 28 |

Depth-by-proposal interaction: `+0.0748` nats entropy-AUC, 95% CI [-0.0060, +0.1582].

## Mechanics

- terminal_cell_failures: `0`.
- all_actions_legal: `True`.
- initial_root_cell_shared: `True`.
- width_call_allocation_matches_virtual_depth_two: `True`.
- physical_llm_requests: `34778`.
- accepted_llm_cells: `34778`.
- raw_rejected_responses: `0`.
- logical_llm_calls: `36000`.
- cache_hits: `1222`.
- inner_llm_calls_used_only_for_action_proposals: `True`.
- grid_score_width_candidates_per_nonterminal_decision: `72`.
