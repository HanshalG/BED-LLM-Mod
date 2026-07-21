# Rock Branch-Strategy Serving Smoke

This is an engineering-only interface and proposal-quality probe. It executes no paired policy trajectories.

- Passed: `True`.
- Parse rate: `10/10`.
- Raw rejected attempts: `0`.
- Every cell has move and check roots: `True`.
- Every cell includes a move-then-check policy: `True`.

| Map | State | Horizon | Position | Best root | Exhaustive fraction | Move / check policies | Move then check |
| --- | ---: | ---: | --- | --- | ---: | --- | ---: |
| 11-11 | 0 | 2 | [0, 5] | move-NORTH | 1.000 | 3 / 3 | 3 |
| 11-11 | 1 | 2 | [1, 5] | move-NORTH | 1.000 | 3 / 3 | 3 |
| 11-11 | 2 | 2 | [0, 5] | move-NORTH | 0.982 | 3 / 3 | 3 |
| 11-11 | 3 | 2 | [0, 5] | move-SOUTH | 1.000 | 3 / 3 | 3 |
| 11-11 | 4 | 1 | [1, 5] | check-3 | 1.000 | 0 / 6 | 0 |
| 11-11 | 5 | 2 | [1, 5] | move-NORTH | 1.000 | 3 / 3 | 3 |
| 11-11 | 6 | 2 | [0, 5] | move-EAST | 0.083 | 3 / 3 | 3 |
| 11-11 | 7 | 2 | [0, 5] | move-NORTH | 1.000 | 3 / 3 | 3 |
| 11-11 | 8 | 2 | [0, 6] | move-NORTH | 0.013 | 3 / 3 | 3 |
| 11-11 | 9 | 1 | [0, 6] | check-1 | 1.000 | 0 / 6 | 0 |

Usage: `{"adapter_completion_tokens": 3265, "adapter_cost_usd": 0.034161750000000005, "adapter_prompt_tokens": 25959, "adapter_reasoning_tokens": 0, "adapter_requests": 10, "backend": "openrouter", "budget_usd": 40.0, "completion_tokens": 3265, "forced_exits": 0, "model": "openai/gpt-5.4-mini", "model_usage": {"openai/gpt-5.4-mini": {"completion_tokens": 3265, "cost_usd": 0.034161750000000005, "prompt_tokens": 25959, "reasoning_tokens": 0, "requests": 10}}, "prompt_tokens": 25959, "reasoning_tokens": 0, "remaining_usd": 15.340297732539984, "requests": 10, "run_budget_usd": 4.0, "run_cost_usd": 0.034161750000000005, "run_remaining_usd": 3.96583825, "total_spent_usd": 24.659702267460016}`.
