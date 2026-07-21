# RockSample[11,11] Exact Follow-Up Closure Screen

This is a zero-LLM-call proposal-fidelity diagnostic, not a policy endpoint.

| Root source | States | Original fraction | Closed fraction | Gain | Optimal-root coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| Gemma StrategyEIG | 330 | 0.7390 | 0.9686 | +0.2296 | 0.9606 |
| Matched random | 330 | 0.2304 | 0.9669 | +0.7366 | 0.9333 |

LLM-minus-random closed fraction: `+0.0017`.

- closed_fraction_at_least_0_97: **False**.
- fraction_gain_at_least_0_08: **True**.
- optimal_root_coverage_at_least_0_90: **True**.
- llm_minus_random_closed_fraction_at_least_0_03: **False**.

**Gate passed: False.**
