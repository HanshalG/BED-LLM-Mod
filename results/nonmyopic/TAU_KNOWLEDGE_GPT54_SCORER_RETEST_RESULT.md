# tau-Knowledge GPT-5.4 Scorer Test-Retest Result

## Decision

The same-task reproducibility gate failed. Semantic root and continuation
ranking reproduced strongly, but only one of three fresh scorer executions
passed the frozen end-to-end endpoint conjunction. No fourth repeat or
threshold repair follows.

## Mechanics

- Replicates completed: `3/3`.
- Physical requests: exact `420/420`.
- Reasoning tokens: `0`.
- Forced exits: `0`.
- Malformed or repaired responses: `0`.
- Tree, query, retrieval, or belief regeneration calls: `0`.
- Adapter-attributed cost: `$1.9437555`, below the `$6.75` cap.
- Live OpenRouter balance: `$50.332023786` before and `$49.100666286`
  afterward, leaving `$24.100666286` above the protected reserve.

## Ranking Reproducibility

All frozen aggregate ranking and agreement gates passed:

- mean non-myopic root pairwise accuracy: `.6777`;
- mean non-myopic-minus-myopic root accuracy: `+.0992`;
- mean focused continuation pairwise accuracy: `.7434`;
- non-myopic root argmax agreement across replicate pairs: `.6333`; and
- focused continuation argmax agreement: `.7867`.

For context, the original V3.1 confirmation reported root accuracy `.7025`,
root gain `+.1488`, and continuation accuracy `.7566`. The three retest root
accuracies were `.6364`, `.6860`, and `.7107`; continuation accuracies were
`.7544`, `.7259`, and `.7500`. The semantic ranking links therefore remain
consistently above the frozen `.60` thresholds.

## Endpoint Reproducibility

| Replicate | Non-myopic docs | Myopic docs | Gain | Original gates |
|---|---:|---:|---:|---|
| 1 | 29 | 26 | +3 | fail |
| 2 | 31 | 25 | +6 | pass |
| 3 | 31 | 29 | +2 | fail |

Only `1/3` replicates achieved both endpoint total at least `25` and gain at
least `+4`; the gate required `2/3`. Replicates 1 and 3 failed only
`end_to_end_total_advantage_over_myopic_at_least_4`. Their win/loss/tie counts
were both `4/2/14`; replicate 2 was `6/1/13`.

Non-myopic endpoint totals were narrow (`29` to `31`), while freshly rescored
myopic totals varied from `25` to `29`. Across repeats the raw totals were
`91` versus `80`, a mean advantage of `+3.67` documents, but repeated scores
on the same 20 tasks are not independent task samples and do not satisfy the
preregistered robustness criterion.

## Interpretation

The result separates two claims. GPT-5.4 reproducibly ranks root strategies and
continuations better than the freshly rescored myopic link on these frozen
trees. The discrete end-to-end advantage over the myopic policy is more
sensitive to scorer execution noise near argmax boundaries and is not
test-retest robust under the frozen `+4` criterion.

This does not erase the original held-out V3.1 result, but it narrows it:
ranking fidelity is the robust evidence; exact endpoint gains remain
directional and execution-sensitive. The test uses the same open tasks, so it
adds no new-domain or new-task generalization evidence, does not establish
cross-model transfer, and does not reverse the refreshed-belief alignment null.

## Artifacts

- Preregistration:
  `results/nonmyopic/TAU_KNOWLEDGE_GPT54_SCORER_RETEST_PREREGISTRATION.md`
- Aggregate:
  `results/nonmyopic/TAU_KNOWLEDGE_GPT54_SCORER_RETEST_ANALYSIS.json`
- Three parsed replicate artifacts:
  `results/nonmyopic/tau_knowledge_gpt54_scorer_retest/`
- Private raw responses: stored outside git.
