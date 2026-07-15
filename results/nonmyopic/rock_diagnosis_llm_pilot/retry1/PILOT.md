# Rock Diagnosis LLM Candidate-Proposal Pilot

**Exploratory only.** The LLM proposes only legal action IDs. Rock dynamics, observations, exact posterior updates, EIG values, action selection, and full-vector MAP decoding are programmatic.

| Arm | Entropy AUC | Final entropy | Final MAP accuracy | Final truth log p | Mean pool | Logical calls / decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| d1_shared | 2.0027 | 1.9216 | 0.1250 | -2.3605 | 3.00 | 1.00 |
| d2 | 1.8594 | 1.5375 | 0.2500 | -1.6327 | 3.00 | 5.17 |
| d1_call_matched_width | 1.9566 | 1.9150 | 0.2500 | -2.1243 | 5.62 | 5.31 |

## Paired Final Entropy

Positive values favor depth two.

| Comparison | Mean entropy reduction | Descriptive 95% bootstrap CI | W / T / L |
| --- | ---: | --- | --- |
| d2_minus_d1_shared | +0.3841 | [+0.2505, +0.5241] | 8 / 0 / 0 |
| d2_minus_d1_call_matched_width | +0.3775 | [+0.2492, +0.5138] | 8 / 0 / 0 |

## Mechanics

- Terminal candidate-cell failures: `0`.
- Raw rejected candidate attempts: `1`.
- All selected actions legal: `True`.
- Initial candidate cells shared: `True`.
- Width call allocation matches virtual d2: `True`.
- Candidate calls: `711` physical / `735` logical; cache hits `24`.

## Decision

- Directional d2 final-entropy gain versus both controls: `True`.
- **Promotable to one preregistered confirmatory run: `True`.**
- This is an eight-task exploratory screen. Its intervals are descriptive, not confirmatory evidence.
