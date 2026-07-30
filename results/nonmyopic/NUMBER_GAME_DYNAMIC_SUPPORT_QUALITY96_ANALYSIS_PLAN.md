# Number Game Dynamic-Support Quality-96 Analysis Plan

Date frozen: 2026-07-30

## Purpose

This is a zero-call retrospective audit of the fresh 96-tree Qwen
dynamic-support study. The source result and its formal status are already
known. This audit cannot rescue, relabel, or replace that result.

The previous mechanism audit established that dynamic support improves
candidate-root ranking and regret, but not why its generated hypotheses are
useful. This audit asks the sharper LLM-native question: does routing generated
hypotheses by simulated answer history make the resulting belief state a
better approximation to the exact canonical posterior than either fixed
support or an equally generated but history-blind support pool?

## Frozen Sources

- source `RESULT.json` SHA256:
  `04177da415498d765baa2b460f6d732a5162b118776df4d5019f7b540bfce36e`
- source `TREES.json` SHA256:
  `8535df7eba5437c564cc6df56816fcee0a6991c3358fdaa6b62c6ca96948da82`
- source `TARGETS.json` SHA256:
  `2fd09b75bcce734e17f237ced9a49e97e13e290e2f5229b9354ad7a8a1bdafd6`
- source tree count: `96`
- candidate roots per tree: `8`
- canonical targets: the exact `33` unique extensions in `TARGETS.json`,
  weighted uniformly

The implementation must fail before producing output if a source hash, tree
count, tree seed, candidate-root set, selected-root record, canonical-target
count, or canonical-target extension hash differs.

## Frozen Supports

All supports are deduplicated by complete extension. Every comparison follows
the dynamic policy's stored path so query history is held fixed.

For canonical target `theta` and candidate root `r`:

1. `y1 = theta(r)`.
2. The second query is the greedy EIG query selected from the stored dynamic
   first-stage support for `(r, y1)`.
3. `y2` is the canonical target's answer to that second query.

At the first stage:

- `fixed`: initial hypotheses consistent with `(r, y1)`;
- `dynamic`: the stored retained-rejuvenation support for `(r, y1)`;
- `blind_pool`: initial hypotheses plus generated first-stage hypotheses from
  both answer branches under `r`, filtered for `(r, y1)` only after pooling.

At the second stage:

- `fixed`: initial hypotheses consistent with `(r, y1, q2, y2)`;
- `dynamic`: the stored retained-rejuvenation support for the realized
  `(r, y1, q2, y2)` branch;
- `blind_pool`: initial hypotheses, all generated first-stage hypotheses under
  `r`, and all generated second-stage hypotheses under `r`, pooled without
  their branch assignment and then filtered by the realized two-answer
  history.

The history-blind pool uses the exact hypotheses and model calls already made
for the tree. It is a nondeployable diagnostic control, not a policy. It has
weakly greater raw opportunity than the routed dynamic support, so a dynamic
advantage cannot be attributed to fewer calls or a wider candidate pool.

## Frozen Metrics

For each support and canonical history, form the uniform-support posterior
predictive probability for every unqueried number. The exact reference is the
uniform posterior over the 33 canonical targets consistent with that history.

Report:

- posterior-predictive mean squared error against the exact canonical
  posterior, averaged equally over canonical targets, then roots, then trees;
- exact truth-extension coverage for each support;
- support size;
- first- and second-stage results separately;
- paired complete-tree bootstrap intervals for all mean support differences.

For each root, define second-stage refresh quality gain as:

`fixed predictive MSE - dynamic predictive MSE`.

On trees where dynamic and fixed support selected different roots, report:

- quality gain at the dynamic-selected root minus quality gain at the
  fixed-selected root;
- its paired tree-bootstrap interval;
- Spearman correlation between that quality-gain contrast and the already
  recorded exact-canonical realized Brier advantage;
- a bootstrap interval for the correlation.

Also report dynamic-versus-blind differences at the dynamic-selected and
fixed-selected roots. Coverage is descriptive because retained union makes
dynamic coverage weakly monotone relative to fixed support by construction.

## Inference

- Bootstrap seed: `87000`.
- Bootstrap samples: `20000`.
- Resampling unit: one complete tree.
- Intervals: percentile 95% intervals.
- Empty approximate support, if encountered, receives predictive MSE `1.0`,
  zero coverage, and size zero.
- Exact canonical posteriors must be nonempty for every evaluated history.
- Spearman ties use the existing repository implementation.
- No multiplicity-adjusted confirmatory claim is made.

The support-quality mechanism is called directionally coherent only if all
three conditions hold:

1. all-root second-stage dynamic-minus-fixed predictive MSE has an interval
   below zero;
2. all-root second-stage dynamic-minus-history-blind predictive MSE has an
   interval below zero; and
3. on changed-root trees, the mean dynamic-root-minus-fixed-root refresh
   quality gain has an interval above zero.

The quality-gain-to-realized-advantage correlation is an explanatory
calibration diagnostic and cannot override these conditions.

## Output

The implementation will write one public `RESULT.json` under:

`results/nonmyopic/number_game_dynamic_support_quality96/`

It makes zero model calls and costs `$0`.
