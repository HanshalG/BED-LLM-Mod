# RockSample[7,8] Exact Follow-Up Closure Screen

This is a zero-LLM-call proposal-fidelity diagnostic, not a policy endpoint.

| Root source | States | Original fraction | Closed fraction | Gain | Optimal-root coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| Gemma StrategyEIG | 270 | 0.8576 | 0.9963 | +0.1387 | 0.8963 |
| Matched random | 270 | 0.2690 | 0.9344 | +0.6653 | 0.8889 |

LLM-minus-random closed fraction: `+0.0619`.

- closed_fraction_at_least_0_97: **True**.
- fraction_gain_at_least_0_08: **True**.
- optimal_root_coverage_at_least_0_90: **False**.
- llm_minus_random_closed_fraction_at_least_0_03: **True**.

**Gate passed: False.**
