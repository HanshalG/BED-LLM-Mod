# Number Game Predictive-Risk Independent-Tree Replication V2 Result

Date: 2026-07-28

Protocols:

- `NUMBER_GAME_PREDICTIVE_RISK_REPLICATION_PREREGISTRATION.md`
- `NUMBER_GAME_PREDICTIVE_RISK_REPLICATION_V2_PREREGISTRATION.md`

Run:
`results/nonmyopic/number_game_predictive_risk_replication_v2/number-game-predictive-risk-replication-v2-20260728T133000Z`

## Verdict

**Strong replication versus myopic, fixed-depth-two, and random controls;
formal conjunctive null because the PTS superiority gate fails.**

All eight fresh planning trees and eight fresh cross-model target supports
completed. Every transport and structural gate passed. Predictive-risk BED
selected a different first query from both myopic EIG and classical
fixed-support depth two on seven of eight trees.

## Aggregate Results

| Baseline | Candidate Brier | Baseline Brier | Relative gain | Brier tree wins | Whole-tree 95% difference CI |
|---|---:|---:|---:|---:|---:|
| myopic EIG | 0.18632 | 0.22094 | 15.67% | 7/8 | [-0.04852, -0.02012] |
| fixed-support depth two | 0.18632 | 0.22146 | 15.87% | 7/8 | [-0.04881, -0.02070] |
| exact uniform random root | 0.18632 | 0.20371 | 8.54% | 8/8 | [-0.02196, -0.01256] |
| seeded PTS | 0.18632 | 0.19176 | 2.84% | 5/8 | [-0.01487, 0.00248] |

Versus myopic EIG, predictive-risk BED also:

- lowers best-rule Hamming error from `0.14032` to `0.09732`
  (`30.65%`);
- has whole-tree Hamming-difference CI `[-0.06607, -0.02077]`;
- wins Hamming on seven of eight trees; and
- improves mean exact-extension coverage by `8.16` percentage points.

On target extensions novel to their planning support, it beats myopic on Brier
in seven of eight trees and on Hamming in six of eight. Mean differences are
`-0.04411` Brier and `-0.04934` Hamming.

## PTS Boundary

The preregistration required at least 5% Brier improvement and a wholly
negative whole-tree interval versus PTS. Observed improvement is only `2.84%`,
with interval `[-0.01487, 0.00248]`. Those two gates fail, so the overall
registered status is `replication_failed`.

This does not erase the independently replicated comparison to myopic BED:
all gates against myopic, fixed-depth-two, and random controls pass with large
margins. It does show that PTS is an unusually strong generator-stability
baseline. Predictive risk beats PTS on five trees, loses on three, and is
statistically unresolved at eight trees.

## Mechanism

The root decision is genuinely proposal-aware. Myopic and fixed-depth-two
select root zero on most trees, while predictive-risk BED chooses roots such as
68, 27, 60, 9, and 61 after simulating the branch-conditioned LLM support.
All policies then use the same generated support after their realized root.
The gain therefore comes from anticipating the LLM belief state induced by the
first query, not from a stronger likelihood or a privileged endpoint update.

The remaining gap is Monte Carlo prior mismatch. On every tree the one-support
planning estimate predicts the chosen root will beat the two PTS roots, but
three independent GPT target supports reverse that ordering. A prospectively
fixed multi-draw proposal prior is the appropriate next development step;
threshold relaxation or dropping PTS is not.

## Accounting

- Accepted requests / HTTP attempts: `144 / 144`
- Retries / provider-error retries: `0 / 0`
- Reasoning tokens / forced exits: `0 / 0`
- Cost: `$0.3709795`
- Result SHA-256:
  `0796310076a9cafd8c8aa0cba6e6c1adef7461d7b88e6c9dac203409e38c8855`
- Tree artifact SHA-256:
  `8817a10f1c121261524508c3b95699e489d072ceb9485bedf54915eaada9473f`
- Private raw-response SHA-256:
  `71f2a5b2d0c003b0536117d8cee2f390131e0beb1003f0e311ca609a98c6aa51`
