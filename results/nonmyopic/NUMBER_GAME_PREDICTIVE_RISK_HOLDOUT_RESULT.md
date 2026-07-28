# Number Game Predictive-Risk Holdout Result

Date: 2026-07-28

Protocol:
`results/nonmyopic/NUMBER_GAME_PREDICTIVE_RISK_HOLDOUT_PREREGISTRATION.md`

Run:
`results/nonmyopic/number_game_predictive_risk_holdout/number-game-predictive-risk-holdout-20260728T123000Z`

## Verdict

**All frozen holdout gates pass. Independent proposal-tree replication is
authorized.**

The policy was frozen from the already-open Gemini planning tree. It selected
root 34 by minimizing two-query terminal posterior-predictive Brier risk over
the current particles. Myopic EIG and classical fixed-support depth two both
selected root 48. No target information changed those roots.

One fresh GPT-5.4 nonreasoning call then generated 24 valid unique executable
target concepts. Twelve target extensions were absent from the Gemini planning
support.

## Fresh Cross-Model Targets

| Policy | Root | Mean Brier | Best-rule Hamming | Exact-extension coverage |
|---|---:|---:|---:|---:|
| predictive Bayes risk | 34 | 0.17599 | 0.06271 | 12/24 |
| myopic EIG | 48 | 0.20519 | 0.11221 | 11/24 |
| fixed-support depth two | 48 | 0.20519 | 0.11221 | 11/24 |
| uniform over eight candidate roots | - | 0.21543 | 0.10618 | 10.75/24 |

Versus myopic/fixed root 48, predictive-risk root 34:

- lowers mean Brier by `14.23%`;
- has paired candidate-minus-baseline Brier CI
  `[-0.05622, -0.00657]`;
- lowers mean best-rule Hamming error by `44.12%`; and
- improves exact-extension coverage by `4.17` percentage points.

Versus the exact uniform-root control, it lowers Brier by `18.31%` with paired
CI `[-0.05799, -0.02150]`.

On the 12 target extensions novel to the planning support, Brier is `12.09%`
lower and Hamming error is `31.34%` lower than myopic. The novel-target Brier
interval crosses zero and novel coverage is lower, so this is a directional
transfer diagnostic, not a standalone claim.

## Interpretation

This is the first clean positive in the project where the LLM's proposal
dynamics are load-bearing. Both policies execute the same deterministic
Number Game likelihood and use the same regenerated supports after their
chosen first query. The difference is that the non-myopic policy simulates how
each root changes the LLM-generated belief state and minimizes terminal
predictive loss. A fixed-support planner cannot calculate that counterfactual
without the branch-conditioned LLM proposals.

The evidence is still one planning tree. The preregistration therefore permits
only an independent multi-tree replication, not a final general claim.

## Accounting

- Accepted requests / HTTP attempts: `1 / 1`
- Retries / reasoning tokens / forced exits: `0 / 0 / 0`
- Cost: `$0.0077275`
- Result SHA-256:
  `59b6368ee98f2240bde6293a5522ed499d32519687374ce950edd8149314af84`
- Target artifact SHA-256:
  `50bb235cc1d5af14e51f478af0bd3113410e18c13096914738392a39a495d909`
- Private raw-response SHA-256:
  `c7ff1f831d794bb2f500af3a468bfc01033cedc10548a0196a7e34a15d07dfee`
