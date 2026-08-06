# Number Game Two-Draw Diversity Bonus Audit

Date completed: 2026-08-06

## Decision

**Reject draw disagreement as an uncertainty penalty; advance one frozen
draw-diversity bonus to a new prospective test.**

This is a zero-call retrospective analysis. It does not change any source
study's status or establish a confirmatory result.

## What Was Measured

Every retained-support depth-three candidate root has six generated future
branch contexts: two after the first answer and four after the second. For
each context, the audit reconstructs the valid canonical extension sets from
the two independently seeded Qwen draws and computes their Jaccard distance.
The root statistic is the mean distance across its six contexts.

The shared initial draw is excluded because it cannot distinguish roots. Raw
response text remains private; the audit publishes only numeric summaries and
binds the private artifacts by SHA-256.

## Diagnostic

Draw disagreement is high:

| Cohort | Trees | Mean branch Jaccard distance |
|---|---:|---:|
| Prior development | 96 | 0.7553 |
| Later fresh cohort | 32 | 0.7721 |

But it has the opposite sign from a reliability warning. Within-tree centered
Jaccard distance correlates negatively with realized Brier (`-0.165` and
`-0.059`): roots whose two draws cover more different hypotheses tend to do
slightly better. Positive coefficients that penalize disagreement worsen root
selection in both cohorts.

## Frozen Retrospective Selector

The best development-grid coefficient was:

```text
z(predicted depth-three Brier) - 0.5 * z(mean draw Jaccard distance)
```

Lower is selected. Applied retrospectively:

| Cohort | Bonus Brier | Original d3 | Reduction | Root changes | W/T/L |
|---|---:|---:|---:|---:|---:|
| Prior 96 | 0.102145 | 0.103332 | 1.15% | 26 | 15/70/11 |
| Later fresh 32 | 0.098885 | 0.100505 | 1.61% | 12 | 8/20/4 |

The effect direction replicates, but uncertainty remains honest:

- prior 95% interval: `[-0.003261, +0.000459]`;
- later-fresh interval: `[-0.004060, +0.000902]`;
- combined source-stratified interval: `[-0.002946, +0.000089]`.

The later cohort was generated after the development cohort, but this audit
is not claimed as prospectively held out because its endpoints already
existed when the analysis was designed.

## Planning Horizon

The larger and more relevant result is monotonic depth:

| Cohort | Bonus d3 | Dynamic d2 | Reduction | 95% paired interval |
|---|---:|---:|---:|---:|
| Prior 96 | 0.102145 | 0.107588 | 5.06% | [-0.008226, -0.002808] |
| Later fresh 32 | 0.098885 | 0.104935 | 5.76% | [-0.011727, -0.000930] |

The bonus also beats myopic EIG by `12.46%` and `17.10%`, and compute-matched
fixed-support depth three by `4.55%` and `4.42%`, with all four cohort-level
intervals below zero.

This suggests the planner was not simply missing more samples. Independent
LLM draws create useful semantic breadth, and treating that breadth as a
planning feature repairs some depth-three mis-ranking. The coefficient is
still retrospectively selected, so only a new frozen fresh-tree cohort can
support the headline claim.

## Next Test

The prospective protocol freezes the coefficient at `-0.5`, makes no sweep,
and tests bonus depth three versus dynamic depth two as the primary endpoint.
It fits one `$5` Europe/London daily allocation and runs only after the sealed
August 7 history-blind control.

## Artifact

Result directory:

`results/nonmyopic/number_game_two_draw_diversity_bonus_audit/number-game-two-draw-diversity-bonus-audit-20260806T003000Z`

- model calls / cost: `0` / `$0`;
- trees: `128` across two hash-bound cohorts;
- bootstrap samples: `20,000`;
- `RESULT.json` SHA-256:
  `49ef14d369a7afe1e530b4cfcb377ee1b2cc280ef4af758fe8f59fd0ecb4caea`;
- private raw response text is not published.
