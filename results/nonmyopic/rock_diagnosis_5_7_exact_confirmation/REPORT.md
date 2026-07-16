# Rock Diagnosis Exact Confirmation

## Scope

This is a zero-LLM-call exact mechanism confirmation. The static target is the full rock-type vector; motion, likelihoods, exact Bayesian updates, and MAP decodes are deterministic/auditable. The held-out map is Figure 4's `5-7` layout from Araya-Lopez, Buffet, and Thomas (2013).

## Final Entropy

Positive reductions favor depth two. Width uses the same number of current-state proposal cells as the full depth-two root tree's root plus outcome branches.

| K | d2 - shared d1 | 95% CI | d2 - call-matched width | 95% CI | Gate |
| ---: | ---: | --- | ---: | --- | --- |
| 3 | +0.0960 | [+0.0866, +0.1058] | +0.0629 | [+0.0534, +0.0728] | pass |

## Mechanics

| K | Shared root cells | Width contains base | Width calls match virtual d2 cells | Legal actions |
| ---: | --- | --- | --- | --- |
| 3 | True | True | True | True |

## Decision

- **Confirmation passes:** `True`.
- Rule: at least one K has d2 entropy superiority over shared d1 and call-matched width.

## Reproduction

```bash
python scripts/nonmyopic_rock_diagnosis_oracle.py
pytest -q tests/test_nonmyopic_rock_diagnosis_oracle.py
```
