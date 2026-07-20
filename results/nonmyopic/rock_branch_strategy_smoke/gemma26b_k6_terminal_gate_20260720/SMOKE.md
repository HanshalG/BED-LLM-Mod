# Rock Branch-Strategy Serving Smoke

This is an engineering-only interface and proposal-quality probe. It executes no paired policy trajectories.

- Passed: `True`.
- Parse rate: `10/10`.
- Raw rejected attempts: `0`.
- Every cell has move and check roots: `True`.
- Every cell includes a move-then-check policy: `True`.

| Map | State | Horizon | Position | Best root | Exhaustive fraction | Move / check policies | Move then check |
| --- | ---: | ---: | --- | --- | ---: | --- | ---: |
| 3-6 | 0 | 2 | [0, 3] | move-EAST | 1.000 | 3 / 3 | 3 |
| 3-6 | 1 | 2 | [1, 3] | move-SOUTH | 1.000 | 3 / 3 | 3 |
| 3-6 | 2 | 2 | [0, 3] | move-EAST | 1.000 | 3 / 3 | 3 |
| 3-6 | 3 | 2 | [0, 3] | move-EAST | 1.000 | 3 / 3 | 3 |
| 3-6 | 4 | 1 | [1, 3] | check-1 | 1.000 | 3 / 3 | 0 |
| 5-7 | 0 | 2 | [0, 3] | move-EAST | 1.000 | 3 / 3 | 3 |
| 5-7 | 1 | 2 | [1, 3] | move-EAST | 1.000 | 3 / 3 | 3 |
| 5-7 | 2 | 2 | [0, 3] | move-EAST | 1.000 | 3 / 3 | 3 |
| 5-7 | 3 | 2 | [0, 3] | move-EAST | 1.000 | 3 / 3 | 3 |
| 5-7 | 4 | 1 | [1, 3] | check-2 | 1.000 | 1 / 5 | 0 |

Usage: `{"adapter_completion_tokens": 4490, "adapter_cost_usd": 0.0029407500000000002, "adapter_prompt_tokens": 15083, "adapter_reasoning_tokens": 0, "adapter_requests": 10, "backend": "openrouter", "budget_usd": 40.0, "completion_tokens": 4490, "forced_exits": 0, "model": "google/gemma-4-26b-a4b-it", "model_usage": {"google/gemma-4-26b-a4b-it": {"completion_tokens": 4490, "cost_usd": 0.0029407500000000002, "prompt_tokens": 15083, "reasoning_tokens": 0, "requests": 10}}, "prompt_tokens": 15083, "reasoning_tokens": 0, "remaining_usd": 19.959468534539972, "requests": 10, "run_budget_usd": 0.25, "run_cost_usd": 0.0029407500000000002, "run_remaining_usd": 0.24705925, "total_spent_usd": 20.040531465460028}`.
