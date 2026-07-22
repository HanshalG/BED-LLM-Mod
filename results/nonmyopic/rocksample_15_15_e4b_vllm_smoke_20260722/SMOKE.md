# Rock Branch-Strategy Serving Smoke

This is an engineering-only interface and proposal-quality probe. It executes no paired policy trajectories.

- Passed: `True`.
- Parse rate: `10/10`.
- Raw rejected attempts: `2`.
- Every cell has move and check roots: `True`.
- Every cell includes a move-then-check policy: `True`.

| Map | State | Horizon | Position | Best root | Exhaustive fraction | Move / check policies | Move then check |
| --- | ---: | ---: | --- | --- | ---: | --- | ---: |
| 15-15 | 0 | 2 | [0, 7] | check-10 | 0.002 | 2 / 2 | 2 |
| 15-15 | 1 | 2 | [1, 7] | move-EAST | 0.000 | 2 / 2 | 2 |
| 15-15 | 2 | 2 | [0, 7] | check-3 | 0.192 | 2 / 2 | 2 |
| 15-15 | 3 | 2 | [0, 7] | check-3 | 0.192 | 2 / 2 | 2 |
| 15-15 | 4 | 1 | [1, 7] | check-0 | 0.000 | 3 / 1 | 0 |
| 15-15 | 5 | 2 | [1, 7] | move-EAST | 0.018 | 2 / 2 | 2 |
| 15-15 | 6 | 2 | [0, 7] | check-1 | 0.096 | 2 / 2 | 2 |
| 15-15 | 7 | 2 | [0, 7] | check-3 | 0.192 | 2 / 2 | 2 |
| 15-15 | 8 | 2 | [0, 8] | check-13 | 0.078 | 2 / 2 | 2 |
| 15-15 | 9 | 1 | [0, 8] | check-12 | 0.000 | 3 / 1 | 0 |

Usage: `{"backend": "vllm", "completion_tokens": 3896, "cost_usd": 0.0, "forced_exits": 0, "model": "google/gemma-4-E4B-it", "model_usage": {"google/gemma-4-E4B-it": {"completion_tokens": 3896, "cost_usd": 0.0, "prompt_tokens": 52060, "reasoning_tokens": 0, "requests": 12}}, "prompt_tokens": 52060, "reasoning_tokens": 0, "requests": 12, "run_cost_usd": 0.0}`.
