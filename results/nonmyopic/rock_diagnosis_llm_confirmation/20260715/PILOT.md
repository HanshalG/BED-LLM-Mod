# Rock Diagnosis LLM Candidate-Proposal Pilot

**Pre-registered confirmatory run.** The LLM proposes only legal action IDs. Rock dynamics, observations, exact posterior updates, EIG values, action selection, and full-vector MAP decoding are programmatic.

| Arm | Entropy AUC | Final entropy | Final MAP accuracy | Final truth log p | Mean pool | Logical calls / decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| d1_shared | 1.9865 | 1.8895 | 0.2000 | -1.8318 | 3.00 | 1.00 |
| d2 | 1.8875 | 1.6886 | 0.2000 | -1.6889 | 3.00 | 5.27 |
| d1_call_matched_width | 1.9396 | 1.8789 | 0.2667 | -1.7881 | 5.62 | 5.29 |

## Paired Final Entropy

Positive values favor depth two.

| Comparison | Mean entropy reduction | 95% bootstrap CI | W / T / L |
| --- | ---: | --- | --- |
| d2_minus_d1_shared | +0.2009 | [+0.0740, +0.3330] | 19 / 0 / 11 |
| d2_minus_d1_call_matched_width | +0.1903 | [+0.0666, +0.3167] | 19 / 0 / 11 |

## Mechanics

- Terminal candidate-cell failures: `0`.
- Raw rejected candidate attempts: `6`.
- All selected actions legal: `True`.
- Initial candidate cells shared: `True`.
- Width call allocation matches virtual d2: `True`.
- Candidate calls: `2684` physical / `2774` logical; cache hits `90`.

## Decision

- Paired final-entropy 95% intervals exclude zero against both controls: `True`.
- **Confirmed under the preregistered criterion: `True`.**
