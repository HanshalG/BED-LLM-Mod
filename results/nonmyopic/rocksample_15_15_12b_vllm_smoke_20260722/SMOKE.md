# Rock Branch-Strategy Serving Smoke

This is an engineering-only interface and proposal-quality probe. It executes no paired policy trajectories.

- Passed: `True`.
- Parse rate: `10/10`.
- Raw rejected attempts: `0`.
- Every cell has move and check roots: `True`.
- Every cell includes a move-then-check policy: `True`.
- Proposal-quality gate enabled: `True`; passed: `True`.
- Mean h2 best/exhaustive fraction: `0.4216555098186036`.
- H2 cells at threshold: `7`.

| Map | State | Horizon | Position | Best root | Exhaustive fraction | Move / check policies | Move then check |
| --- | ---: | ---: | --- | --- | ---: | --- | ---: |
| 15-15 | 0 | 2 | [0, 7] | check-3 | 0.233 | 2 / 2 | 2 |
| 15-15 | 1 | 2 | [1, 7] | move-EAST | 1.000 | 2 / 2 | 2 |
| 15-15 | 2 | 2 | [0, 7] | check-3 | 0.233 | 2 / 2 | 2 |
| 15-15 | 3 | 2 | [0, 7] | move-EAST | 0.308 | 2 / 2 | 2 |
| 15-15 | 4 | 1 | [1, 7] | check-3 | 1.000 | 2 / 2 | 0 |
| 15-15 | 5 | 2 | [1, 7] | check-14 | 0.133 | 2 / 2 | 2 |
| 15-15 | 6 | 2 | [0, 7] | check-3 | 0.233 | 2 / 2 | 2 |
| 15-15 | 7 | 2 | [0, 7] | check-3 | 0.233 | 2 / 2 | 2 |
| 15-15 | 8 | 2 | [0, 8] | move-EAST | 1.000 | 2 / 2 | 2 |
| 15-15 | 9 | 1 | [0, 8] | check-4 | 1.000 | 2 / 2 | 0 |

Usage: `{"backend": "vllm", "completion_tokens": 3120, "cost_usd": 0.0, "forced_exits": 0, "model": "google/gemma-4-12B-it", "model_usage": {"google/gemma-4-12B-it": {"completion_tokens": 3120, "cost_usd": 0.0, "prompt_tokens": 41976, "reasoning_tokens": 0, "requests": 10}}, "prompt_tokens": 41976, "reasoning_tokens": 0, "requests": 10, "run_cost_usd": 0.0}`.
