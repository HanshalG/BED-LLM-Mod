# Bongard Classical-Horizon Opportunity Stratum Protocol

Frozen: 2026-08-09, before the first paid Bongard response and before any
mechanics, development, confirmation, or endpoint outcome.

## Question

The random-task primary can be null for two different reasons: the regenerated
LLM planner may fail, or many sampled tasks may offer little horizon-dependent
action opportunity. This protocol separates those explanations without using a
Luna response or a true candidate/endpoint label.

## Endpoint-Blind Stratum

For each frozen task, compare the first query selected by each already-frozen
classical planner with the first query selected by that planner's own myopic
counterpart:

- DINOv2 depth two versus DINOv2 myopic;
- SigLIP depth two versus SigLIP myopic.

`classical_horizon_disagreement` is the union: at least one encoder changes its
first query under depth two. `classical_horizon_agreement` is the complement.
The definition uses only the frozen plans and their model-predicted utilities.
It does not read true candidate labels, endpoint labels, Luna responses, or any
realized policy metric.

Frozen inputs:

- DINOv2 `PLANS.json` SHA-256
  `58154f052424f1632c302f9b40af030c172fc4d626908c9f977211c1e8299847`;
- DINOv2 `MANIFEST.json` SHA-256
  `56e9d538ded501f4518dc662d8403f3f15ba254c8cedb132e4bd09d10949e504`;
- SigLIP `PLANS.json` SHA-256
  `a41cc3b01d18fa9008f67d6f60a3f113b8f3194b73e932b4e2c3cfc83215f587`;
- SigLIP `MANIFEST.json` SHA-256
  `95c00d948ee77589995c3434e3eb01b0a908b496fe2d984b859dbae4032d057d`.

The frozen plan hashes imply:

| Partition | Total | Disagreement | Agreement |
|---|---:|---:|---:|
| mechanics | 4 | 1 | 3 |
| development | 64 | 27 | 37 |
| confirmation | 96 | 49 | 47 |

## Analysis

After a stage result is independently authorized and replayed, report within
each stratum:

- dynamic versus `compute_matched_myopic_ensemble` first-query and final-history
  changes;
- paired mean-Brier and mean-log-loss differences with the same 20,000-draw
  bootstrap convention used by the primary analysis;
- dynamic and compute-matched score-to-realized-utility Spearman means;
- dynamic relative Brier gain against the compute-matched control.

Negative paired differences favor dynamic depth two. The analysis must verify
the exact compute-matched score contract before accepting a tree.

## Claim Boundary

This is a prospectively frozen secondary diagnostic. It changes no random-task
primary gate, claim tier, confirmation authorization, task, prompt, seed,
model, request, endpoint, or budget. It makes zero model calls and cannot rescue
a failed primary result. Any future opportunity-enriched execution requires a
separate prospective amendment and tested fail-closed runner before it is
authorized.
