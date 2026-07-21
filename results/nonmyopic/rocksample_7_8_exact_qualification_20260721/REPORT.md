# Rock Diagnosis Exact Confirmation

## Scope

This is a zero-LLM-call exact mechanism confirmation. The static target is the full rock-type vector; motion, likelihoods, exact Bayesian updates, and MAP decodes are deterministic/auditable. The held-out map is `7-8` from Smith and Simmons (2004), page 5.

## Final Entropy

Positive reductions favor depth two. Width uses the same number of current-state proposal cells as the full depth-two root tree's root plus outcome branches.

| K | d2 - shared d1 | 95% CI | d2 - call-matched width | 95% CI | Gate |
| ---: | ---: | --- | ---: | --- | --- |
| 12 | +1.6775 | [+1.6568, +1.6980] | +1.6775 | [+1.6567, +1.6986] | pass |

## Mechanics

| K | Shared root cells | Width contains base | Width calls match virtual d2 cells | Legal actions |
| ---: | --- | --- | --- | --- |
| 12 | True | True | True | True |

## Decision

- **Confirmation passes:** `True`.
- Rule: at least one K has d2 entropy superiority over shared d1 and call-matched width.

## Reproduction

```bash
python scripts/nonmyopic_rock_diagnosis_oracle.py
pytest -q tests/test_nonmyopic_rock_diagnosis_oracle.py
```
