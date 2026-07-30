# Number Game Dynamic-Support Mechanism-96 Result

Date: 2026-07-30

## Status

**Directionally coherent retrospective mechanism audit.** All three frozen
diagnostic conditions pass. This analysis makes zero model calls and does not
rescue, relabel, or replace the source study's mechanics-qualified
`gated_null` status.

## Candidate-Ranking Fidelity

Across the same eight candidate roots on each of 96 fresh trees:

| Metric | Dynamic support | Fixed support | Dynamic minus fixed |
|---|---:|---:|---:|
| Mean Spearman | 0.31796 | 0.15703 | 0.16093 |
| Pairwise concordance | 0.61830 | 0.56111 | 0.05720 |

The paired tree-bootstrap intervals for the differences are:

- Spearman: `[0.07853, 0.24456]`;
- concordance: `[0.02557, 0.08935]`.

Answer-conditioned support regeneration therefore improves ranking of the
complete candidate set against exact-canonical realized Brier, rather than
only changing the selected root.

## Changed-Root First Link

Dynamic and fixed support select different roots on `72/96` trees. On those
trees:

- mean exact-canonical realized advantage is `0.004907` Brier;
- the 95% tree-bootstrap interval is `[0.001295, 0.008446]`;
- dynamic records `50/0/22` wins/ties/losses.

The other 24 trees select the same root, explaining the source study's overall
`50/24/22` record.

The dynamic simulated margin has Spearman `0.207` with realized advantage, but
its interval `[-0.031, 0.431]` crosses zero. The fixed counter-margin is
uncorrelated (`-0.010`), and the combined rank-reversal margin is `0.209` with
an interval crossing zero. Thus selection direction is resolved, while
tree-to-tree calibration of the predicted advantage magnitude remains noisy.

## Candidate-Set Oracle Regret

The diagnostic oracle chooses the lowest exact-canonical Brier root among the
same eight candidates:

| Metric | Dynamic support | Fixed support |
|---|---:|---:|
| Mean oracle regret | 0.008537 | 0.012217 |
| Exact oracle-root selections | 27/96 | 18/96 |

Dynamic minus fixed regret is `-0.003680`, with 95% interval
`[-0.006388, -0.000973]`. This is an explanatory oracle, not a deployable
policy.

## Support Novelty

Across `4,608` generated branches, `73.91%` of generated extension hashes are
novel relative to the answer-consistent parent support, averaging `19.46`
novel hypotheses per branch.

The dynamic-selected roots do not simply have more novelty:

- combined novel count: `19.55` dynamic versus `20.30` fixed;
- combined novel fraction: `73.92%` versus `74.24%`;
- dynamic-minus-fixed novelty correlations with realized advantage are only
  `0.10`--`0.11`.

The mechanism is therefore better described as **path-conditioned support
quality changing query valuation**, not as a raw hypothesis-count effect.

## Interpretation

This audit closes the most important explanatory gap in the powered result.
The dynamic-support simulator:

1. ranks all candidate roots better than the recomputed fixed-support
   simulator;
2. selects changed roots with positive realized Brier advantage; and
3. reduces exact candidate-set oracle regret.

It does not establish precise calibration of simulated margin magnitude, and
it cannot repair the three frozen support-floor misses or the adverse Hamming
diagnostic in the source study.

## Provenance

- source `RESULT.json` SHA256:
  `04177da415498d765baa2b460f6d732a5162b118776df4d5019f7b540bfce36e`
- source `TREES.json` SHA256:
  `8535df7eba5437c564cc6df56816fcee0a6991c3358fdaa6b62c6ca96948da82`
- audit `RESULT.json` SHA256:
  `53ae668d14073759604fc2f19b97a05babfc521820517eab1017f18a5ed205a5`
- bootstrap seed/samples: `86000` / `20000`
- model calls / cost: `0` / `$0`
