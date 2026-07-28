# Number Game Ranking-Fidelity Audit Result

Date: 2026-07-28

This is a post-hoc, zero-call diagnostic over the three independent fresh
32-tree Number Game datasets. It evaluates the first link in the planning
chain: whether current-particle simulated terminal risk ranks roots in the same
order as independently generated held-out target rules.

## Method

Each public tree has eight common candidate roots, a current LLM-generated
particle support, root-conditioned branch supports, and an independently
generated target support. For every root, the audit compares:

1. simulated terminal posterior-predictive Brier risk under the current
   particles; and
2. realized terminal Brier and Hamming endpoints on the held-out targets.

It reports within-tree Spearman correlation and pairwise concordance, the
selected root's held-out regret relative to the per-tree oracle, and root
stability after deleting each current particle in turn. Myopic EIG and
fixed-support depth-two scores are converted to risks before ranking.

## Ranking Fidelity

| Fresh dataset | Predictive-risk Spearman Brier | Pairwise concordance | Myopic Spearman | Oracle / top-2 roots | Leave-one-particle-out agreement |
|---|---:|---:|---:|---:|---:|
| GPT-mini planner / Gemini targets | 0.754 [0.659, 0.830] | 0.820 | -0.045 | 14 / 21 of 32 | 0.821 |
| Gemini planner / GPT targets, confirmation | 0.719 [0.634, 0.791] | 0.798 | -0.293 | 11 / 24 of 32 | 0.874 |
| Gemini planner / GPT targets, powered | 0.709 [0.607, 0.791] | 0.799 | -0.381 | 20 / 23 of 32 | 0.910 |
| Combined descriptive | 0.727 [0.675, 0.774] | 0.806 [0.784, 0.827] | -0.239 [-0.311, -0.167] | 45 / 68 of 96 | 0.868 [0.835, 0.901] |

The combined fixed-support depth-two Spearman correlation is also negative:
`-0.237` with tree-bootstrap interval `[-0.309, -0.163]`. Predictive-risk
Spearman correlation with held-out Hamming is `0.460`
(`[0.398, 0.520]`).

## Oracle Regret

| Policy | Mean held-out Brier regret | Candidate minus baseline | 95% tree-bootstrap CI | Candidate wins |
|---|---:|---:|---:|---:|
| Predictive-risk BED | 0.00704 | - | [0.00519, 0.00906] | - |
| Myopic EIG | 0.04487 | -0.03783 | [-0.04615, -0.03015] | 84/96 |
| Fixed-support depth two | 0.04743 | -0.04039 | [-0.04879, -0.03282] | 85/96 |
| Positive-test strategy | 0.03365 | -0.02661 | [-0.03288, -0.02088] | 83/96 |
| Uniform random candidate | 0.03391 | -0.02687 | [-0.03052, -0.02335] | 90/96 |

The predictive-risk root is the held-out oracle on 45/96 trees, lies in the
top two on 68/96, and has mean endpoint rank `2.094`.

## Particle Stability

Mean agreement between the full-support root and roots selected after deleting
one particle is `86.84%` (`[83.49%, 90.08%]`). Forty of 96 roots are unchanged
under every deletion, and 70/96 retain at least 75% agreement. The ranking
signal is therefore not driven by a single indispensable particle.

## Interpretation

The proposal-aware simulator's ordering is strongly aligned with independent
target performance in every dataset and under a planning/target model-family
swap. In contrast, immediate EIG and fixed-support depth-two scores are
anti-aligned in the combined data. This directly validates the first link
needed for non-myopic selection: the LLM-generated current support predicts
which root-conditioned semantic proposal process will perform well.

This does not prove perfect simulation or monotonic gains with arbitrary
planning depth. It is a descriptive reuse of open endpoints, uses a restricted
executable-rule grammar, and does not repair the depth-three support failures.

Public result SHA-256:
`735935bbaaca7b08016fc0745160f9ba1d2d68b9d9ce8d089ca6866a83ec2e11`
