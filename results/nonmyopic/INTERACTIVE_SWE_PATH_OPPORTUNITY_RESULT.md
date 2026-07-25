# Interactive SWE Path-Dependent Opportunity Result

## Decision

The frozen zero-call opportunity gate failed. This exact clarification wrapper
is closed without an LLM smoke, threshold repair, probe diversification, cohort
selection, or development/holdout access.

The released benchmark remains scientifically attractive, but the fixed
target-blind probes collapse to too few distinct hidden-issue answers. Paying an
LLM to rank this tree would mostly compare roots with identical observations.

## Frozen Results

| Metric | Result | Gate |
|---|---:|---:|
| Opportunity rows | 120 | exactly 120 |
| Usable rows | 29 | at least 60 |
| Strict non-myopic opportunities | 4 | at least 18 |
| Repositories with a strict opportunity | 3 | at least 6 |
| Mean normalized depth-two gain | `.0338511` | at least `.025` |
| Strict rows with immediate sacrifice and final gain | 4/4 | all |
| Mean distinct root answers | `2.0167` | descriptive |

Source, split, endpoint ordering, mean-gain, and strict-row mechanics passed.
The three prevalence/diversity gates failed.

## Bottleneck

Target availability was not the main limitation:

- 81/120 rows had at least five hidden gold-change tokens.
- Only 35/120 had at least three distinct root answers.
- Only 29/120 satisfied both conditions.
- Root-answer counts were: 44 rows with one, 41 with two, 25 with three,
  nine with four, and one with five.

Among the 29 usable rows, depth two changed the greedy root only four times.
All four changes were strict improvements, so the exact tree is coherent where
it differs; it is simply too sparse for a powered LLM comparison.

The four strict rows were:

| Instance | Repository | Immediate greedy -> d2 | Final greedy -> d2 |
|---|---|---:|---:|
| `django__django-13410` | Django | `3 -> 2` | `4 -> 13` |
| `scikit-learn__scikit-learn-25102` | scikit-learn | `6 -> 5` | `10 -> 12` |
| `sympy__sympy-15349` | SymPy | `2 -> 0` | `2 -> 3` |
| `scikit-learn__scikit-learn-14894` | scikit-learn | `9 -> 7` | `10 -> 11` |

The mean normalized-gain gate is partly driven by the large Django gain. It
does not compensate for only four strict decisions.

## Integrity

- Public audit SHA-256:
  `e1767b2ae445b0462666f1f8356d1843f227a5098303b4c895a914fb1c393bb3`.
- Endpoint columns loaded only after all target-blind trees froze.
- Development 40 and holdout 330 hidden fields remain unused.
- OpenRouter calls and cost: `0 / $0`.
- OatML use: none.

No paper claim is added. The result is a prospective environment-screening
null and leaves the existing tau-Knowledge result as the strongest LLM-native
evidence.
