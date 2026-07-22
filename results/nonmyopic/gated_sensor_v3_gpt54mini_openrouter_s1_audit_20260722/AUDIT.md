# Gated Sensor Continuation-Fidelity Diagnostic

This zero-LLM-call audit separates fixed-root coverage from branch-continuation quality.

| Arm | h2 states | Continuation efficiency | Root coverage | Proposal / exhaustive d2 | Optimal continuation |
| --- | ---: | ---: | ---: | ---: | ---: |
| LLM StrategyEIG | 28 | 0.6938 | 1.0000 | 0.6938 | 0.2500 |
| Matched random on LLM states | 28 | 0.7834 | 1.0000 | 0.7834 | 0.2857 |
| Reached random arm | 28 | 0.7982 | 1.0000 | 0.7982 | 0.2857 |

LLM-minus-random continuation efficiency: `-0.0896`.
LLM-minus-random proposal/exhaustive fraction: `-0.0896`.
