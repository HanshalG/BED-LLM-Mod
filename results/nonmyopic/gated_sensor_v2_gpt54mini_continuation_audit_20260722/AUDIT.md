# Gated Sensor Continuation-Fidelity Diagnostic

This zero-LLM-call audit separates fixed-root coverage from branch-continuation quality.

| Arm | h2 states | Continuation efficiency | Root coverage | Proposal / exhaustive d2 | Optimal continuation |
| --- | ---: | ---: | ---: | ---: | ---: |
| LLM StrategyEIG | 56 | 0.7591 | 1.0000 | 0.7591 | 0.3214 |
| Matched random on LLM states | 56 | 0.8020 | 1.0000 | 0.8020 | 0.3571 |
| Reached random arm | 56 | 0.8182 | 1.0000 | 0.8182 | 0.3214 |

LLM-minus-random continuation efficiency: `-0.0429`.
LLM-minus-random proposal/exhaustive fraction: `-0.0429`.
