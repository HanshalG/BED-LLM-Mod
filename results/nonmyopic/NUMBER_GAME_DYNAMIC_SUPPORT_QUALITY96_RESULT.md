# Number Game Dynamic-Support Quality-96 Result

Date: 2026-07-30

## Status

**Frozen directional null with a positive selective-quality diagnostic.** None
of the three preregistered directional conditions pass. This retrospective
analysis makes zero model calls and does not rescue, relabel, or replace the
source study's mechanics-qualified `gated_null` status.

## Canonical Posterior Approximation

The audit replays all 33 uniformly weighted canonical concepts through every
candidate root and compares each approximate support's posterior-predictive
probabilities with the exact canonical posterior.

| Stage | Fixed MSE | Dynamic MSE | Dynamic minus fixed 95% CI |
|---|---:|---:|---:|
| First refresh | 0.01808 | 0.03914 | `[0.01898, 0.02313]` |
| Second refresh | 0.03313 | 0.03289 | `[-0.00188, 0.00137]` |

The first refresh worsens global canonical posterior approximation. The
second-stage difference is nearly zero (`-0.000244`) and its interval crosses
zero. Dynamic support is therefore not a globally better posterior
approximation.

## Truth Coverage

Dynamic support substantially expands exact truth-extension coverage:

| Stage | Fixed | Dynamic | Difference |
|---|---:|---:|---:|
| First refresh | 0.4091 | 0.5493 | +0.1402 |
| Second refresh | 0.4091 | 0.6181 | +0.2090 |

Mean support size grows from `18.22` to `37.08` after one answer and from
`9.72` to `40.52` after two. Coverage is descriptive, because retained union
makes it weakly monotone relative to filtered initial support by construction.
The result shows the tradeoff clearly: generation recovers more possible
truths while initially diluting predictive calibration.

## Selective Refresh Quality

Dynamic and fixed support select different roots on `72/96` trees. Define a
root's refresh-quality gain as fixed minus dynamic second-stage canonical
predictive MSE.

- mean gain at the dynamic-selected root: `0.005810`;
- mean gain at the fixed-selected root: `0.003045`;
- mean contrast: `0.002766`;
- contrast 95% interval: `[-0.002718, 0.008106]`.

The mean contrast is imprecise and fails the frozen third gate. However, its
tree-level Spearman correlation with exact-canonical realized Brier advantage
is `0.472`, with bootstrap interval `[0.262, 0.645]`. This was frozen as a
calibration diagnostic, not a pass condition. It supports a selective
mechanism: dynamic planning helps when its chosen root is one where
answer-conditioned refresh improves the belief approximation.

## History-Blind Control

The same-call history-blind pool is exactly identical to routed dynamic
support on every evaluated metric. This is structural rather than evidence
that prompt conditioning is irrelevant. Parsing already requires every
generated hypothesis to satisfy the branch observations; after all branches
are pooled, consistency filtering removes hypotheses from counterfactual
answers and recovers the routed support exactly.

Consequently, the second frozen condition fails at equality and this stored
artifact cannot identify the effect of conditioning the **generation prompt**
on history. A genuinely informative control must make fresh history-blind
generations under matched calls and then filter them against each branch.

## Frozen Gates

| Condition | Result |
|---|---|
| Second-stage dynamic MSE CI below fixed | Fail |
| Second-stage dynamic MSE CI below history-blind pool | Fail at exact equality |
| Changed-root mean refresh-quality contrast CI above zero | Fail |

The formal support-quality mechanism is therefore not directionally coherent.
The positive quality-gain correlation is retained as an explanatory
diagnostic, not promoted into a post-hoc gate.

## Provenance

- source `RESULT.json` SHA256:
  `04177da415498d765baa2b460f6d732a5162b118776df4d5019f7b540bfce36e`
- source `TREES.json` SHA256:
  `8535df7eba5437c564cc6df56816fcee0a6991c3358fdaa6b62c6ca96948da82`
- source `TARGETS.json` SHA256:
  `2fd09b75bcce734e17f237ced9a49e97e13e290e2f5229b9354ad7a8a1bdafd6`
- audit `RESULT.json` SHA256:
  `139ff20ebc4cc056c6145192e335598eca5dd2ccb9f0a18aad54c47518bfbd7e`
- deterministic replay: exact SHA256 match
- bootstrap seed/samples: `87000` / `20000`
- model calls / cost: `0` / `$0`
