# tau-Knowledge Future-First Robustness Preregistration

Date: 2026-07-26

This zero-call audit tests whether the future-uplift decomposition already
supported on two disjoint tau-Knowledge splits also stabilizes policy selection
across six physically independent GPT-5.4 scorer executions on the fixed V3.1
trees. No new model call, retrieval, tree generation, task, or endpoint is
introduced.

## Frozen Inputs

The exact V3.1 confirmation artifact has SHA-256:

`f8b120b0d2b98f9f10eb0ef9251262a6a41b20d4d90d724ee4926c141ec275ae`

The scorer-retest block contains:

1. `932ca6c046a0c8bd6cd9c48786600e549da3f3dd10e4eb80ddf540b52b8f4135`
2. `9c81e75209c095f508f3838ac4da72bf6b5693aa2924f0d005c00e4009f72877`
3. `4ef14acb5121629510d887c18414d9881f19cc7e15503365ac105013026fc4ec`

The rank-ensemble confirmation block contains:

1. `d28a863fa4d6fce30aa2cecbba9c65c8c7c46613620db87d55bca1becec5c953`
2. `0cccf93296ac336b2ea587f7206cb22059cfd789c27aab075aa6add9f3a106f6`
3. `607adc749723bdd5490a111989d839432f1aa56a192a0b554af49913e4ba726a`

Every replicate must contain the same ordered 20 task IDs, five root scores per
task, five four-way focused continuation vectors per task, and 100 exact root
diagnostics whose pair values reproduce the source artifact.

## Frozen Policies

For task \(t\), root \(r\), myopic score \(m_{tr}\), and full-tree score
\(f_{tr}\), define:

\[
u_{tr}=f_{tr}-m_{tr}.
\]

The **future-first** root maximizes the lexicographic tuple
\((u_{tr},m_{tr},-r)\). Its continuation maximizes that replicate's focused
continuation scores for the selected root, breaking ties by index.

Controls are:

- **raw full:** maximize \(f_{tr}\), then root order, with the same focused
  continuation rule;
- **myopic:** maximize \(m_{tr}\), then root order, with the same focused
  continuation rule;
- **random:** use the already-frozen random root and continuation recorded by
  each replicate.

No score normalization, weighting, clipping, ensemble, task filter, or
confidence fallback is allowed.

## Frozen Measurements

From the source artifact, immediate root value is the number of required
documents retrieved by the first query. Total root value is the best exact
required-document count among its four continuations. Future-only value is
total minus immediate.

For each replicate report:

- pairwise accuracy of \(u\) against exact future-only value;
- future-first root selection and cross-replicate root agreement;
- exact 20-task endpoint totals for future-first, raw full, myopic, and random;
- future-first wins, ties, and losses against each control.

Across all six runs report both three-run block means, pooled pairwise accuracy,
mean endpoint totals, and the number of replicate-level nonnegative
differences. For inference, average each future-first-minus-control endpoint
difference over the six runs within each task, then use an exact one-sided
task-level sign-flip test over the 20 task averages. Zero differences are
removed.

## Frozen Gates

A **strong robustness pass** requires all of:

1. both three-run blocks have mean future-gain pairwise accuracy at least
   `.55`;
2. pooled future-gain pairwise accuracy is at least `.60`;
3. mean pairwise agreement of the six future-first root vectors is at least
   `.50`;
4. mean 20-task endpoint gain is at least `+2` documents versus myopic;
5. mean endpoint gain is at least `+1` document versus raw full;
6. future-first is nonnegative in at least four of six replicate totals versus
   both myopic and raw full;
7. the task-clustered exact one-sided sign-flip value versus myopic is at most
   `.10`.

A **directional result** requires both block accuracies at least `.50`, pooled
accuracy at least `.55`, positive mean endpoint gain versus both myopic and raw
full, and at least three of six nonnegative replicate totals against both.
Anything else is null or adverse.

## Interpretation Boundary

These trees, tasks, and endpoints are already open. A pass would show that an
LLM-derived future-uplift signal is reproducible across independent scoring
executions and can stabilize a fixed semantic policy. It would not establish
new-task generalization, repair the failed fresh execution, or replace a
prospective external confirmation. A failure closes this future-first
stabilization rule on the V3.1 tree bank.
