# Rock Diagnosis Exact-Scorer Scaling

One exact scorer unit is one evaluated action node in the depth-two policy tree; it is not a wall-clock or hidden-state likelihood operation.

| Map | Hidden states | K | StrategyEIG units / decision | Exhaustive d2 units / decision | Exhaustive / StrategyEIG | LLM calls / decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 3-6 | 8 | 6 | 13.67 | 64.25 | 4.70x | 1.0 |
| 5-7 | 32 | 6 | 13.87 | 115.12 | 8.30x | 1.0 |
| 7-8 | 128 | 6 | 14.10 | 209.40 | 14.85x | 1.0 |
| 11-11 | 2,048 | 6 | 14.24 | 354.17 | 24.88x | 1.0 |
| 15-15 | 32,768 | 4 | 9.60 | 616.94 | 64.26x | 1.0 |

Registered bounded-K StrategyEIG keeps exact action-tree width small while exhaustive d2 expands every legal root and continuation. The relative node reduction grows monotonically from 4.70x to 64.26x across the confirmed maps. Every StrategyEIG decision uses one proposal call and exact rollout scoring uses no LLM calls.

The first four runs use K6; the preregistered 15-rock run uses K4 after K4 matched K6 endpoint quality in the 11-rock width study. This is observed registered-budget scaling, not a fixed-K causal comparison.

These ratios isolate action-tree work. Both methods still evaluate likelihoods over the full hidden-state vector, so they are not wall-clock speedups.
