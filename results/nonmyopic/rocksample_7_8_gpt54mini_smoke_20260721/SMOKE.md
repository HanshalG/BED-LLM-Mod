# Rock Branch-Strategy Serving Smoke

This is an engineering-only interface and proposal-quality probe. It executes no paired policy trajectories.

- Passed: `True`.
- Parse rate: `10/10`.
- Raw rejected attempts: `0`.
- Every cell has move and check roots: `True`.
- Every cell includes a move-then-check policy: `True`.

| Map | State | Horizon | Position | Best root | Exhaustive fraction | Move / check policies | Move then check |
| --- | ---: | ---: | --- | --- | ---: | --- | ---: |
| 7-8 | 0 | 2 | [0, 3] | move-SOUTH | 1.000 | 3 / 3 | 3 |
| 7-8 | 1 | 2 | [1, 3] | move-EAST | 1.000 | 3 / 3 | 3 |
| 7-8 | 2 | 2 | [0, 3] | move-SOUTH | 1.000 | 3 / 3 | 3 |
| 7-8 | 3 | 2 | [0, 3] | move-SOUTH | 1.000 | 3 / 3 | 3 |
| 7-8 | 4 | 1 | [1, 3] | check-2 | 1.000 | 0 / 6 | 0 |
| 7-8 | 5 | 2 | [1, 3] | move-EAST | 1.000 | 3 / 3 | 3 |
| 7-8 | 6 | 2 | [0, 3] | move-EAST | 0.431 | 3 / 3 | 3 |
| 7-8 | 7 | 2 | [0, 3] | move-SOUTH | 1.000 | 3 / 3 | 3 |
| 7-8 | 8 | 2 | [0, 4] | move-SOUTH | 1.000 | 3 / 3 | 3 |
| 7-8 | 9 | 1 | [0, 4] | check-2 | 0.029 | 0 / 6 | 0 |

Usage: `{"adapter_completion_tokens": 3321, "adapter_cost_usd": 0.029040750000000004, "adapter_prompt_tokens": 18795, "adapter_reasoning_tokens": 0, "adapter_requests": 10, "backend": "openrouter", "budget_usd": 40.0, "completion_tokens": 3321, "forced_exits": 0, "model": "openai/gpt-5.4-mini", "model_usage": {"openai/gpt-5.4-mini": {"completion_tokens": 3321, "cost_usd": 0.029040750000000004, "prompt_tokens": 18795, "reasoning_tokens": 0, "requests": 10}}, "prompt_tokens": 18795, "reasoning_tokens": 0, "remaining_usd": 19.355077832539973, "requests": 10, "run_budget_usd": 4.0, "run_cost_usd": 0.029040750000000004, "run_remaining_usd": 3.97095925, "total_spent_usd": 20.644922167460027}`.
