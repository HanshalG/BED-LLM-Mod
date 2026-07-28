# Number Game First-Retention Path-Dependence Audit

Date: 2026-07-28

Status: **passed zero-call mechanism audit**

## Question

The fresh retained depth-three experiment preserved generated-only support
after the first answer and retained parent particles only after the second.
Its remaining proposed change is to retain consistent initial particles at the
first refresh as well. This audit asks:

1. Does first retention materially change the next query selected by greedy
   EIG?
2. Does it recover independent target rules that first-step generation
   dropped?
3. Can the resulting full-retention experiment be reconstructed from existing
   second-step responses, or are fresh LLM calls necessary?

This is a post-hoc mechanism audit, not a policy endpoint comparison.

## Exact Method

For every root/first-answer branch in the open eight-tree development dataset
and the independent fresh six-tree confirmation:

- parse the initial and generated first-step executable-rule supports;
- union the generated support with consistent initial particles, deduplicated
  by exact extension over `0..100`;
- reproduce the recorded greedy second query on generated-only support;
- select the counterfactual greedy second query on retained support; and
- measure exact-extension coverage on the independently generated target
  rules whose first answer follows that branch.

Every operation is deterministic and local. There are no model calls. Trees
are the bootstrap unit with 50,000 resamples.

## Results

| Source | Trees | Query changes | Old-response reuse | Generated coverage | Retained coverage | Coverage gain |
|---|---:|---:|---:|---:|---:|---:|
| Development | 8 | 84/128 (65.6%) | 34.4% | 22.9% | 50.6% | +27.6 pp |
| Fresh confirmation | 6 | 75/96 (78.1%) | 21.9% | 21.6% | 53.6% | +32.0 pp |
| Combined | 14 | 159/224 (71.0%) | 29.0% | 22.4% | 51.8% | +29.5 pp |

The combined tree-bootstrap interval is `[63.4%, 77.7%]` for the second-query
change rate and `[25.3, 33.1]` percentage points for target-coverage recovery.
All 14 trees have at least one changed future query. Retention recovers 738 of
2,504 target paths and loses none.

Generated first supports average 18.04 rules and have minimum size 7.
Retained supports average 26.34 and have minimum size 16. On retained support,
switching from the recorded query to the newly greedy query gains a mean
`0.01494` nats of immediate EIG.

All 224 recorded second queries are exactly reproduced from their original
generated-only supports. This is the key integrity check: query changes come
from the support intervention, not from a scoring mismatch.

## Interpretation

First-step LLM regeneration is strongly path-dependent in the decision-relevant
sense. It drops many current particles that remain consistent with the answer;
restoring them more than doubles independent-target support coverage and
changes most next actions.

Only 65/224 branches retain the old second query. Therefore 159 counterfactual
second-query branches have no matching response in the existing datasets.
The powered full-retention experiment cannot be evaluated honestly by replay
or endpoint rescoring; fresh branch-conditioned LLM generation is required.

The audit does not show that full-retention depth three beats depth two.
It establishes that the proposed intervention changes the LLM belief
transition and future policy path at substantial scale, rather than merely
adding inert particles. The frozen fresh 20-tree experiment remains the policy
test.

## Integrity

- Development `TREES.json` SHA-256:
  `016ed9218e9f034984e0745c9a7cb62b4111901db821c065e20c38c5ce76b65f`
- Fresh confirmation `TREES.json` SHA-256:
  `b6c157f7b4d15b4934a2329adeba7c0c28557293aab48de201f5ec8aba0b4f3c`
- Audit `RESULT.json` SHA-256:
  `4ec136a2bb71137564d108aac919a262892b1f3696a34e1358dac8a73035cea3`
- Model calls and cost: `0`

Artifact:
`results/nonmyopic/number_game_first_retention_path_dependence_audit/RESULT.json`
