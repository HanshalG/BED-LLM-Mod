# UCI Zoo LLM Candidate-Proposal Pilot

**Exploratory only.** The frozen matrix supplies every answer, likelihood, posterior update, EIG score, and MAP decode. The LLM only proposes legal trait IDs.

| Arm | Accuracy AUC | Final MAP accuracy | Final entropy | Mean unique pool | Logical calls / decision |
| --- | ---: | ---: | ---: | ---: | ---: |
| d1_shared | 0.0417 | 0.1250 | 1.3170 | 3.00 | 1.00 |
| d2 | 0.0625 | 0.1250 | 1.2617 | 3.00 | 5.31 |
| d1_matched_width | 0.0417 | 0.1250 | 1.2207 | 13.12 | 5.48 |

## Paired Accuracy-AUC

| Comparison | Mean delta | Descriptive 95% bootstrap CI | W / T / L |
| --- | ---: | --- | --- |
| d2_minus_d1_shared | +0.0208 | [+0.0000, +0.0625] | 1 / 7 / 0 |
| d2_minus_d1_matched_width | +0.0208 | [+0.0000, +0.0625] | 1 / 7 / 0 |

## Mechanics

- No invalid LLM candidate responses: `False`.
- Initial candidate cells shared across all arms: `True`.
- Width allocation equals its virtual depth-two allocation at every decision: `True`.
- Candidate requests: `494` physical / `566` logical; cache hits `72`.

## Decision

- Directional d2 gain versus both controls: `True`.
- **Promotable to one preregistered confirmatory run: `False`.**
- This is an eight-task exploratory screen. The intervals are descriptive and are not confirmatory evidence.
