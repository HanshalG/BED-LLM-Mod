# Rock Branch-Strategy Serving Smoke

This is an engineering-only interface and proposal-quality probe. It executes no paired policy trajectories.

- Passed: `True`.
- Parse rate: `10/10`.
- Raw rejected attempts: `0`.
- Every cell has move and check roots: `True`.
- Every cell includes a move-then-check policy: `True`.

| Map | State | Horizon | Position | Best root | Exhaustive fraction | Move / check policies | Move then check |
| --- | ---: | ---: | --- | --- | ---: | --- | ---: |
| 11-11 | 0 | 2 | [0, 5] | move-NORTH | 1.000 | 2 / 2 | 2 |
| 11-11 | 1 | 2 | [1, 5] | move-NORTH | 1.000 | 2 / 2 | 2 |
| 11-11 | 2 | 2 | [0, 5] | move-NORTH | 0.982 | 2 / 2 | 2 |
| 11-11 | 3 | 2 | [0, 5] | move-NORTH | 0.982 | 2 / 2 | 2 |
| 11-11 | 4 | 1 | [1, 5] | check-0 | 0.190 | 3 / 1 | 0 |
| 11-11 | 5 | 2 | [1, 5] | move-NORTH | 1.000 | 2 / 2 | 2 |
| 11-11 | 6 | 2 | [0, 5] | move-NORTH | 1.000 | 2 / 2 | 2 |
| 11-11 | 7 | 2 | [0, 5] | move-NORTH | 1.000 | 2 / 2 | 2 |
| 11-11 | 8 | 2 | [0, 6] | check-1 | 0.105 | 2 / 2 | 2 |
| 11-11 | 9 | 1 | [0, 6] | check-1 | 1.000 | 3 / 1 | 0 |

Usage: `{"adapter_completion_tokens": 2683, "adapter_cost_usd": 0.00378013, "adapter_prompt_tokens": 27473, "adapter_reasoning_tokens": 0, "adapter_requests": 10, "backend": "openrouter", "budget_usd": 40.0, "completion_tokens": 2683, "forced_exits": 0, "model": "google/gemma-4-26b-a4b-it", "model_usage": {"google/gemma-4-26b-a4b-it": {"completion_tokens": 2683, "cost_usd": 0.00378013, "prompt_tokens": 27473, "reasoning_tokens": 0, "requests": 10}}, "prompt_tokens": 27473, "reasoning_tokens": 0, "remaining_usd": 11.864650292540013, "requests": 10, "run_budget_usd": 2.0, "run_cost_usd": 0.00378013, "run_remaining_usd": 1.99621987, "total_spent_usd": 28.135349707459987}`.
