# Rock Diagnosis LLM Candidate-Proposal Pilot

**Pre-registered confirmatory run.** The LLM proposes only legal action IDs. Rock dynamics, observations, exact posterior updates, EIG values, action selection, and full-vector MAP decoding are programmatic.

| Arm | Entropy AUC | Final entropy | Final MAP accuracy | Final truth log p | Mean pool | Logical calls / decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| d1_shared | 3.4427 | 3.4142 | 0.0667 | -3.3171 | 3.00 | 1.00 |
| d2 | 3.1280 | 2.8324 | 0.0333 | -2.7961 | 3.00 | 5.11 |
| d1_call_matched_width | 3.4103 | 3.3696 | 0.1000 | -3.2884 | 7.07 | 5.24 |

## Paired Final Entropy

Positive values favor depth two.

| Comparison | Mean entropy reduction | 95% bootstrap CI | W / T / L |
| --- | ---: | --- | --- |
| d2_minus_d1_shared | +0.5819 | [+0.5069, +0.6437] | 29 / 0 / 1 |
| d2_minus_d1_call_matched_width | +0.5372 | [+0.4572, +0.6047] | 29 / 0 / 1 |

## Mechanics

- Terminal candidate-cell failures: `0`.
- Raw rejected candidate attempts: `51`.
- All selected actions legal: `True`.
- Initial candidate cells shared: `True`.
- Width call allocation matches virtual d2: `True`.
- Candidate calls: `2654` physical / `2724` logical; cache hits `70`.

## Decision

- Paired final-entropy 95% intervals exclude zero against both controls: `True`.
- **Confirmed under the preregistered criterion: `True`.**
