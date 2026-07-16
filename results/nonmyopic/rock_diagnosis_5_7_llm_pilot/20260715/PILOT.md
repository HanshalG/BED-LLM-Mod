# Rock Diagnosis LLM Candidate-Proposal Pilot

**Exploratory only.** The LLM proposes only legal action IDs. Rock dynamics, observations, exact posterior updates, EIG values, action selection, and full-vector MAP decoding are programmatic.

| Arm | Entropy AUC | Final entropy | Final MAP accuracy | Final truth log p | Mean pool | Logical calls / decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| d1_shared | 3.4427 | 3.4252 | 0.0000 | -3.4118 | 3.00 | 1.00 |
| d2 | 3.0067 | 2.8196 | 0.0000 | -2.8158 | 3.00 | 5.12 |
| d1_call_matched_width | 3.4074 | 3.3832 | 0.0000 | -3.3708 | 7.09 | 5.30 |

## Paired Final Entropy

Positive values favor depth two.

| Comparison | Mean entropy reduction | Descriptive 95% bootstrap CI | W / T / L |
| --- | ---: | --- | --- |
| d2_minus_d1_shared | +0.6056 | [+0.4968, +0.6732] | 8 / 0 / 0 |
| d2_minus_d1_call_matched_width | +0.5636 | [+0.4488, +0.6488] | 8 / 0 / 0 |

## Mechanics

- Terminal candidate-cell failures: `0`.
- Raw rejected candidate attempts: `11`.
- All selected actions legal: `True`.
- Initial candidate cells shared: `True`.
- Width call allocation matches virtual d2: `True`.
- Candidate calls: `708` physical / `731` logical; cache hits `23`.

## Decision

- Directional d2 final-entropy gain versus both controls: `True`.
- **Promotable to one preregistered confirmatory run: `True`.**
- This is an eight-task exploratory screen. Its intervals are descriptive, not confirmatory evidence.
