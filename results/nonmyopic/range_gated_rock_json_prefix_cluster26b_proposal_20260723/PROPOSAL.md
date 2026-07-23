# Range-Gated Rock Successor-Grounded Proposal Gate

Gate passed: **True**.

| Endpoint | Mean | 95% CI | W/T/L |
| --- | ---: | ---: | ---: |
| LLM - identical-root random | +0.480875 | [+0.479935, +0.481823] | 16/0/0 |
| LLM h3 - shared-plan d2 | +0.485883 | [+0.484944, +0.487119] | 16/0/0 |
| LLM h3 - strong d2 root | +0.479607 | [+0.479607, +0.479607] | 16/0/0 |
| Exact h3 opportunity recovery | +1.000000 | [+1.000000, +1.000000] | 16/0/0 |

- Exact h3 route-root selection: `1.000`.
- Mechanics: `{'sixteen_distinct_cells_resolved': True, 'all_cells_are_strict_d3_opportunities': True, 'all_roots_match_fixed_interface': True, 'all_controls_exactly_scored': True, 'scoring_made_no_llm_calls': True, 'usage_accounted': True}`.
- Endpoint gates: `{'matched_random_lower_bound_positive': True, 'shared_d2_lower_bound_positive': True, 'strong_d2_lower_bound_positive': True, 'route_selection_at_least_threshold': True, 'mean_recovery_at_least_threshold': True}`.
- Usage: `{'backend': 'vllm', 'model': 'google/gemma-4-26B-A4B-it', 'run_cost_usd': 0.0, 'requests': 32, 'prompt_tokens': 117154, 'completion_tokens': 67204, 'reasoning_tokens': 0, 'cost_usd': 0.0, 'model_usage': {'google/gemma-4-26B-A4B-it': {'requests': 32, 'prompt_tokens': 117154, 'completion_tokens': 67204, 'reasoning_tokens': 0, 'cost_usd': 0.0}}, 'forced_exits': 0, 'forced_finalization_events': 16}`.
