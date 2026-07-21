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
| 7-8 | 1 | 2 | [1, 3] | move-NORTH | 1.000 | 3 / 3 | 3 |
| 7-8 | 2 | 2 | [0, 3] | move-SOUTH | 1.000 | 3 / 3 | 3 |
| 7-8 | 3 | 2 | [0, 3] | move-SOUTH | 1.000 | 3 / 3 | 3 |
| 7-8 | 4 | 1 | [1, 3] | check-2 | 1.000 | 0 / 6 | 0 |
| 7-8 | 5 | 2 | [1, 3] | move-NORTH | 1.000 | 3 / 3 | 3 |
| 7-8 | 6 | 2 | [0, 3] | move-SOUTH | 1.000 | 3 / 3 | 3 |
| 7-8 | 7 | 2 | [0, 3] | move-SOUTH | 1.000 | 3 / 3 | 3 |
| 7-8 | 8 | 2 | [0, 4] | move-SOUTH | 1.000 | 3 / 3 | 3 |
| 7-8 | 9 | 1 | [0, 4] | check-5 | 1.000 | 0 / 6 | 0 |

Usage: `{"adapter_completion_tokens": 4525, "adapter_cost_usd": 0.00356748, "adapter_prompt_tokens": 19657, "adapter_reasoning_tokens": 0, "adapter_requests": 10, "backend": "openrouter", "budget_usd": 40.0, "completion_tokens": 4525, "forced_exits": 0, "model": "google/gemma-4-26b-a4b-it", "model_usage": {"google/gemma-4-26b-a4b-it": {"completion_tokens": 4525, "cost_usd": 0.00356748, "prompt_tokens": 19657, "reasoning_tokens": 0, "requests": 10}}, "prompt_tokens": 19657, "reasoning_tokens": 0, "remaining_usd": 19.599873952540012, "requests": 10, "run_budget_usd": 2.0, "run_cost_usd": 0.00356748, "run_remaining_usd": 1.99643252, "total_spent_usd": 20.400126047459988}`.
