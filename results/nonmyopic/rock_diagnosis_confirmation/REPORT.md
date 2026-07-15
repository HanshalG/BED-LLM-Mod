# Rock Diagnosis Exact Confirmation

## Scope

This is a zero-LLM-call exact mechanism confirmation. The static target is the full rock-type vector; motion, likelihoods, exact Bayesian updates, and MAP decodes are deterministic/auditable. The held-out map is Figure 4's `3-6` layout from Araya-Lopez, Buffet, and Thomas (2013).

## Final Entropy

Positive reductions favor depth two. Width uses the same number of current-state proposal cells as the full depth-two root tree's root plus outcome branches.

| K | d2 - shared d1 | 95% CI | d2 - call-matched width | 95% CI | Gate |
| ---: | ---: | --- | ---: | --- | --- |
| 2 | +0.0556 | [+0.0442, +0.0670] | +0.0079 | [-0.0035, +0.0197] | no |
| 3 | +0.2270 | [+0.2129, +0.2414] | +0.1565 | [+0.1417, +0.1709] | pass |
| 4 | +0.4591 | [+0.4464, +0.4716] | +0.4022 | [+0.3890, +0.4151] | pass |

## Mechanics

| K | Shared root cells | Width contains base | Width calls match virtual d2 cells | Legal actions |
| ---: | --- | --- | --- | --- |
| 2 | True | True | True | True |
| 3 | True | True | True | True |
| 4 | True | True | True | True |

## Decision

- **Confirmation passes:** `True`.
- Rule: at least one K has d2 entropy superiority over shared d1 and call-matched width.

## Reproduction

```bash
python scripts/nonmyopic_rock_diagnosis_oracle.py
pytest -q tests/test_nonmyopic_rock_diagnosis_oracle.py
```
