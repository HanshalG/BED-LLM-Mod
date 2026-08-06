# Number Game Two-Draw Diversity Bonus Power Audit

Date completed: 2026-08-06

## Decision

**Use 64 fresh trees in two mandatory 32-tree daily blocks.**

The unopened one-day 32-tree confirmation is underpowered once its monotonic
depth endpoint and conservative bonus non-worsening check are both required.
Ninety-six trees add too little power to justify a third Qwen day under the
current budget.

This is a zero-call retrospective design audit. The selector remains exactly
`z(depth-three risk) - 0.5 * z(two-draw branch Jaccard distance)`. No future
seed, response, or endpoint was opened.

## Estimated Pass Probability

The simulation resampled paired tree rows with replacement from the two
hash-bound retrospective cohorts. Each pool and sample size used 50,000 draws
with seed `108700`.

### Pooled 128-Tree Distribution

| Fresh trees | Primary monotonic endpoint | Primary + non-worsening | Old redundant gate |
|---:|---:|---:|---:|
| 32 | 64.4% | 43.8% | 30.4% |
| 64 | 90.5% | 69.1% | 39.6% |
| 96 | 95.8% | 78.9% | 43.3% |

The primary endpoint requires:

- at least 3% Brier reduction versus dynamic-support depth two;
- an approximate 95% interval below zero; and
- wins exceeding losses.

The joint estimate additionally requires at least one quarter of roots to
change from unadjusted depth three and mean bonus Brier not to be worse.

The old gate also required a scaled `14/32` win floor and changed-root wins to
exceed losses against original depth three. It is redundant with the paired
mean/interval criteria and becomes perversely harder as sample size grows:
its scaled depth-two win-floor pass probability is only `56.9%`, `54.9%`, and
`53.8%` at 32, 64, and 96 trees.

## Cohort Sensitivity

At 64 trees:

| Empirical pool | Primary | Primary + non-worsening |
|---|---:|---:|
| Prior development 96 | 90.3% | 55.8% |
| Later fresh 32 | 82.4% | 79.6% |
| Pooled 128 | 90.5% | 69.1% |

The mechanism check is heterogeneous, but the monotonic depth endpoint is
well powered in both source distributions. It should remain a required
non-worsening direction, not a second significance test.

## Design Consequence

The prospective run will use two fully fresh, mandatory 32-tree blocks on
separate `$5` Europe/London days. Block B is authorized by Block A mechanics
only and runs regardless of Block A scientific values. The combined 64-tree
analysis is the sole confirmatory decision.

The final experiment retains the exact 20,000-sample paired tree bootstrap.
The power audit uses mean plus `1.96` standard errors only as a computationally
tractable approximation for repeated design simulation.

## Artifact

Result directory:

`results/nonmyopic/number_game_two_draw_diversity_bonus_power_audit/number-game-two-draw-diversity-bonus-power-audit-20260806T013000Z`

- `RESULT.json` SHA-256:
  `3dfd1605deba87641c791718a8b50d81e971062fee85612fa94638fc674f156e`;
- model calls / cost: `0` / `$0`.
