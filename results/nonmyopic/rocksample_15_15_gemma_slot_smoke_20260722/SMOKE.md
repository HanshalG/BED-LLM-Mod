# Rock Branch-Strategy Serving Smoke

This is an engineering-only interface and proposal-quality probe. It executes no paired policy trajectories.

- Passed: `True`.
- Parse rate: `10/10`.
- Raw rejected attempts: `0`.
- Every cell has move and check roots: `True`.
- Every cell includes a move-then-check policy: `True`.

| Map | State | Horizon | Position | Best root | Exhaustive fraction | Move / check policies | Move then check |
| --- | ---: | ---: | --- | --- | ---: | --- | ---: |
| 15-15 | 0 | 2 | [0, 7] | move-EAST | 1.000 | 2 / 2 | 2 |
| 15-15 | 1 | 2 | [1, 7] | move-EAST | 1.000 | 2 / 2 | 2 |
| 15-15 | 2 | 2 | [0, 7] | move-EAST | 1.000 | 2 / 2 | 2 |
| 15-15 | 3 | 2 | [0, 7] | check-3 | 0.219 | 2 / 2 | 2 |
| 15-15 | 4 | 1 | [1, 7] | check-0 | 0.000 | 3 / 1 | 0 |
| 15-15 | 5 | 2 | [1, 7] | move-EAST | 1.000 | 2 / 2 | 2 |
| 15-15 | 6 | 2 | [0, 7] | check-3 | 0.219 | 2 / 2 | 2 |
| 15-15 | 7 | 2 | [0, 7] | check-3 | 0.381 | 2 / 2 | 2 |
| 15-15 | 8 | 2 | [0, 8] | move-EAST | 1.000 | 2 / 2 | 2 |
| 15-15 | 9 | 1 | [0, 8] | check-3 | 1.000 | 2 / 2 | 0 |

Usage: `{"adapter_completion_tokens": 2328, "adapter_cost_usd": 0.00438036, "adapter_prompt_tokens": 41984, "adapter_reasoning_tokens": 0, "adapter_requests": 10, "backend": "openrouter", "budget_usd": 40.0, "completion_tokens": 2328, "forced_exits": 0, "model": "google/gemma-4-26b-a4b-it", "model_usage": {"google/gemma-4-26b-a4b-it": {"completion_tokens": 2328, "cost_usd": 0.00438036, "prompt_tokens": 41984, "reasoning_tokens": 0, "requests": 10}}, "prompt_tokens": 41984, "reasoning_tokens": 0, "remaining_usd": 9.76720306254003, "requests": 10, "run_budget_usd": 1.5, "run_cost_usd": 0.00438036, "run_remaining_usd": 1.49561964, "total_spent_usd": 30.23279693745997}`.
