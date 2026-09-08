# Terminal batching clears depth-two runtime

Previous turn produced equivalent cached risk calculations but h2 still exceeded
five seconds. This turn vectorizes all terminal-node conjugate updates and model
weights without creating individual regression states for those leaves.

The common HorizonPlanner has an optional expected_terminal_risk hook. Models
without it retain scalar evaluation. The hook is used only at depth1 or the last
action of a fixed sequence. It returns risk and evaluated leaf count; those leaves
still count against the existing node budget. Reported policy trees continue to
be materialized through scalar branches, so the accelerated objective can be
checked against its displayed tree. No horizon truncation or likelihood binning.

The mixture shares exactly the same merged quadrature nodes/masses between its
scalar and batch paths. It computes each component's continuous density,
posterior parameters and model probability for every node, then sums within and
between-component target variance. A 64MiB estimated batch workspace cap prevents
unbounded dense arrays. This is a workspace estimate, not a total process-memory
guarantee. Batching changes arithmetic order, not physical/model support.

## Verification

156 tests pass in16.89s, including the full SciLaws suite and existing ordinary,
raw and batch chemistry horizon tests; scoped lint passes. New tests cover all
eight public geometries at a conditioned state over every action, full h1/h2/h3
root vectors in adaptive and open-loop modes against a scalar-only wrapper,
materialized risk agreement, invalid hook outputs and preserved node caps.
Comparisons use 1e-12 absolute/relative tolerances. The small depth fixtures do
not qualify scientific multi-step integration or demonstrate source efficacy.

## Isolated public preflight

Pushed implementation 130b9a1c before the changed-version run. No test/profile
process ran concurrently with this preflight. Exact result SHA256:
4eb4c373ab7974e7771537c8aa8dac5f7af1ffd942b39e4ae70f900bf4b91073.

| Public task | h1 seconds | h2 seconds | h2 nodes | h3 |
|---|---:|---:|---:|---|
| Baseball | .025 | 2.330 | 66849 | Node limit |
| Bird flight | .022 | 2.051 | 66849 | Node limit |
| Lake thermocline | .011 | 1.428 | 62481 | Node limit |
| Battery ageing | .010 | 1.409 | 62481 | Node limit |
| Mars craters | .011 | 1.446 | 62481 | Node limit |
| Spirometry | .018 | 1.873 | 66849 | Node limit |
| Volcanic column | .010 | 1.406 | 62481 | Node limit |
| Wind turbine | .010 | 1.403 | 62481 | Node limit |

All8 h2 complete under unchanged five-second/100000-node limits; all8 h3 reach
the node cap. This is materially different from the previous time bottleneck.
An exhaustive third level multiplies both alternative actions and numerical
observations. More low-level speed alone cannot evade a counted-node limit.
Do not repeatedly rerun identical h3 or reinterpret batching as zero leaf work.

The next dependency is an explicit complexity/integration audit: either valid
lower/upper bounds permitting equivalent search pruning, or a prospectively
qualified predictive-integration method with an error criterion and bounded
branching. Neither may discard model hypotheses or modify the source physics,
measurement menu, four-real-query budget or empirical endpoint to get a pass.
The original capped result stays banked; no cap increase is authorized here.

h2 selects a different root from h1 on this generic prior. That is not evidence
of a deployed four-query benefit. Root values for h1 and h2 refer to different
numbers of planned future observations. Tasks of equal dimension again have the
same normalized generic prior. No source outcomes, calibrated LLM proposals or
path-dependent discovery have been tested by these calculations.

All processes exited; no source measurements, model calls or paid cost. Account
and London ledger unchanged, automation paused. Full research goal remains active.
