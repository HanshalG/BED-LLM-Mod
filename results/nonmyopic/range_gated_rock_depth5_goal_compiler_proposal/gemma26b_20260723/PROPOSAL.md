# Range-Gated Rock Hierarchical H5 Proposal Gate

Gate passed: **True**.

| Endpoint | Mean | 95% CI | W/T/L |
| --- | ---: | ---: | ---: |
| LLM goals - random goals | +0.330423 | [+0.180731, +0.450178] | 12/3/1 |
| Compiled h5 - shared compiled h4 | +0.446940 | [+0.387222, +0.477018] | 15/1/0 |
| Compiled h5 - strongest exact d4 root | +0.440652 | [+0.381796, +0.470080] | 15/0/1 |
| Exact h5 opportunity recovery | +0.937398 | [+0.812193, +1.000000] | 15/0/1 |

- Exact h5 route selection: `0.938`.
- Mechanics: `{'sixteen_distinct_cells_resolved': True, 'all_cells_are_strict_h5_opportunities': True, 'all_exact_h5_roots_are_north': True, 'all_registered_routes_are_exact_h5': True, 'all_roots_match_fixed_interface': True, 'all_targets_are_distinct': True, 'all_controls_exactly_scored': True, 'scoring_made_no_llm_calls': True, 'usage_accounted': True}`.
- Endpoint gates: `{'matched_random_goal_lower_bound_positive': True, 'shared_h4_lower_bound_positive': True, 'strong_d4_lower_bound_positive': True, 'route_selection_at_least_threshold': True, 'mean_recovery_at_least_threshold': True}`.
- Usage: `{'backend': 'openrouter', 'model': 'google/gemma-4-26b-a4b-it', 'run_cost_usd': 0.03263011999999999, 'requests': 32, 'prompt_tokens': 47787, 'completion_tokens': 67971, 'reasoning_tokens': 42955, 'model_usage': {'google/gemma-4-26b-a4b-it': {'completion_tokens': 67971, 'cost_usd': 0.03263011999999999, 'prompt_tokens': 47787, 'reasoning_tokens': 42955, 'requests': 32}}, 'total_spent_usd': 40.34322111245983, 'budget_usd': 110.0, 'remaining_usd': 69.65677888754017, 'run_budget_usd': 0.25, 'run_remaining_usd': 0.21736988000000002, 'adapter_cost_usd': 0.03263011999999999, 'adapter_requests': 32, 'adapter_prompt_tokens': 47787, 'adapter_completion_tokens': 67971, 'adapter_reasoning_tokens': 42955, 'forced_exits': 16, 'forced_final_requests': 16, 'forced_final_successes': 16}`.
