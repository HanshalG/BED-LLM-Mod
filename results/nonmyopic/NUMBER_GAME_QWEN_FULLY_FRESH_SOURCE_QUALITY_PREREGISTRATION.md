# Number Game Fully Fresh Source-Quality Diagnostic Preregistration

Date frozen: 2026-08-06, after the source aggregate was opened and before any
source-quality diagnostic output was computed.

## Scope

This is a retrospective, zero-call replay of the existing
`number-game-dynamic-support-quality96-1` analysis on the newly hash-bound
32-tree source cohort. It diagnoses why the source retained a strong
depth-three-over-myopic result while missing the dynamic-versus-fixed-support
conjunction.

It cannot rescue, relabel, or change the source `gated_null`; authorize or
cancel the already-mandatory history-blind control; create a confirmatory
mechanism claim; or select a favorable subset. All 32 trees, eight roots per
tree, 33 canonical targets, and both refresh stages are included.

## Frozen Inputs

- source result SHA-256:
  `13fd3361a8ef8f525f68733182a9bdb30151e37a4a5e14dfb35dde78540ab523`;
- source trees SHA-256:
  `f812e4a356f5129a6f2f22d5f0f995b76b4ac624a0f312154064ff8c51f2c7b0`;
- source targets SHA-256:
  `b799a5d6609f5e8088e2f0115ec1eaa3c4520cebcd187283daf679714cdc5e2b`;
- source artifact commit: `e8e3a23d`;
- tree count: `32`;
- roots per tree: `8`;
- canonical targets: `33`, uniformly weighted;
- bootstrap seed / samples: `100900 / 20000`.

## Unchanged Analysis

For each root and canonical target, replay the target's first label, choose the
dynamic support's exact best second query, replay its second label, and compare
three supports against the exact canonical posterior predictive distribution:

1. fixed: initial support filtered by the realized observations;
2. dynamic: the routed answer-conditioned generated support;
3. stored blind pool: all already-generated root branches pooled and then
   filtered by the realized observations.

At both stages report posterior-predictive MSE, exact truth-extension coverage,
and support size. On changed-root trees report:

- fixed-minus-dynamic second-stage quality gain at the dynamic-selected root;
- the same gain at the fixed-selected root;
- their paired contrast and tree-bootstrap interval;
- Spearman correlation and tree-bootstrap interval between that contrast and
  realized fixed-minus-dynamic root Brier.

The stored blind pool is known from the prior audit to collapse exactly to the
routed support after consistency filtering. It is a structural replay, not the
fresh history-blind generation control scheduled for a later budget day.

There are no pass gates. Signs and intervals are descriptive. The prior
96-tree quality result may be quoted alongside this cohort but is not pooled,
used as a rescue, or treated as independent confirmation of a post-hoc claim.

OpenRouter calls / cost: `0 / $0`. OatML use: `0`.
