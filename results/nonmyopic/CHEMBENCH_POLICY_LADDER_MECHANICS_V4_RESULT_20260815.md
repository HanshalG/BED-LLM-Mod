# ChemBench Policy-Ladder Mechanics V4 Result

Date: 2026-08-15 (Europe/London)

## Status

**Passed and independently verified.** This was the first and only opening of
the prospectively frozen outside-support v4 cohort. It used no LLM/API call and
cost $0.

## Primary Result

| Policy level | Aggregate terminal MSE | Successive reduction |
| --- | ---: | ---: |
| d1 | 0.0391942507 | - |
| d2 | 0.0320363611 | 18.2626% |
| d3 | 0.0292467959 | 8.7075% |

- d2 versus d1: 38 wins, 75 practical ties, 31 losses.
- d3 versus d2: 26 wins, 111 practical ties, 7 losses.
- d2 changes the d1 root on all three slices.
- d3 changes the d2 root on one of three slices.
- Planned terminal Bayes risk equals the mean truth-conditional replay on every
  slice and level within the frozen `1e-10` tolerance.
- Every slice is non-increasing from d1 to d2 to d3.
- Call-matched d1 replay is exact and adds zero producer misses.

All ten conjunctive mechanics conditions pass. The banked-proposer verifier
reconstructs the source responses and exactly reproduces every root, planned
value, truth loss, aggregate comparison, and gate.

## Interpretation

This establishes a real non-myopic opportunity under dynamic M-open support:
successive exact policy-improvement levels reduce untouched-cohort expected
terminal predictive risk while using the same branch-conditioned proposal
transition and experiment budget.

It is not yet the workshop headline. The speculative world prior and proposal
transition are registry oracles, observations are categorical, and each model
uses one fixed parameter state. The result authorizes the next zero-call
proposal-model/likelihood fidelity work and, after that passes, a small LLM
semantic and action-ranking gate. It does not establish LLM efficacy by itself.

## Artifacts

- Result SHA256:
  `f46e1fac28332b06ac2c62ff91fd68e537849c93ea74199ee7dc5d2d418aaec6`
- Transition bank SHA256:
  `eb6c51b945c05f697b25681ed02a49763cb50ea234436b691a9d45a7e69b3b59`
- Verification SHA256:
  `534a2e2173f1919f3f0d20b4d0a921ca6408a52c844fec4e7c4b60a888a3f6e8`
- Required pushed implementation commit: `2f270cbf`.
