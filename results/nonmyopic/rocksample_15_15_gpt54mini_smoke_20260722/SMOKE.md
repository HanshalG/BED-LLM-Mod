# Rock Branch-Strategy Serving Smoke

This is an engineering-only interface and proposal-quality probe. It executes no paired policy trajectories.

- Passed: `True`.
- Parse rate: `10/10`.
- Raw rejected attempts: `0`.
- Every cell has move and check roots: `True`.
- Every cell includes a move-then-check policy: `True`.
- Proposal-quality gate enabled: `False`; passed: `True`.
- Mean h2 best/exhaustive fraction: `0.481682572837813`.
- H2 cells at threshold: `None`.

| Map | State | Horizon | Position | Best root | Exhaustive fraction | Move / check policies | Move then check |
| --- | ---: | ---: | --- | --- | ---: | --- | ---: |
| 15-15 | 0 | 2 | [0, 7] | move-EAST | 0.308 | 2 / 2 | 2 |
| 15-15 | 1 | 2 | [1, 7] | check-3 | 0.431 | 2 / 2 | 2 |
| 15-15 | 2 | 2 | [0, 7] | move-EAST | 0.308 | 2 / 2 | 2 |
| 15-15 | 3 | 2 | [0, 7] | check-3 | 0.192 | 2 / 2 | 2 |
| 15-15 | 4 | 1 | [1, 7] | check-3 | 1.000 | 1 / 3 | 0 |
| 15-15 | 5 | 2 | [1, 7] | move-NORTH | 1.000 | 2 / 2 | 2 |
| 15-15 | 6 | 2 | [0, 7] | move-NORTH | 0.308 | 2 / 2 | 2 |
| 15-15 | 7 | 2 | [0, 7] | move-EAST | 0.308 | 2 / 2 | 2 |
| 15-15 | 8 | 2 | [0, 8] | move-EAST | 1.000 | 2 / 2 | 2 |
| 15-15 | 9 | 1 | [0, 8] | check-3 | 1.000 | 2 / 2 | 0 |

Usage: `{"adapter_completion_tokens": 2450, "adapter_cost_usd": 0.03948225, "adapter_prompt_tokens": 37943, "adapter_reasoning_tokens": 0, "adapter_requests": 10, "backend": "openrouter", "budget_usd": 70.0, "completion_tokens": 2450, "forced_exits": 0, "model": "openai/gpt-5.4-mini", "model_usage": {"openai/gpt-5.4-mini": {"completion_tokens": 2450, "cost_usd": 0.03948225, "prompt_tokens": 37943, "reasoning_tokens": 0, "requests": 10}}, "prompt_tokens": 37943, "reasoning_tokens": 0, "remaining_usd": 39.22821579254004, "requests": 10, "run_budget_usd": 6.0, "run_cost_usd": 0.03948225, "run_remaining_usd": 5.96051775, "total_spent_usd": 30.77178420745996}`.
