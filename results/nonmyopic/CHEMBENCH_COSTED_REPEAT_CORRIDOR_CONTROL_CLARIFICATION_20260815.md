# ChemBench costed-repeat corridor control clarification

Date frozen: 2026-08-15

This clarification is frozen before any v4 response matrix or corridor result is
constructed. It does not change the scientific cohort, observation model, actions,
budget, shortlist, primary policies, or gates in
`CHEMBENCH_COSTED_REPEAT_CORRIDOR_PROTOCOL_20260815.md`.

## Cost-blind control

The cost-blind comparator uses policy depth 3. Its planner treats every repeat
variant as costing one unit when evaluating future action sequences, while
feasibility and realized execution subtract the true repeat cost in `{1, 2, 4}`.
It uses the same prospective beliefs, action shortlist, atomic proposal cache, and
eight-well realized budget as dynamic d3.

## Random control

For each difficulty, evaluate eight deterministic random feasible policies. At
each state, hash the public state, available base assays, remaining true well
budget, and replicate seed, then select uniformly by hash index from all feasible
costed actions in stable identity order. Remove all repeat variants of the selected
base assay and subtract its true repeat cost. Atomic support transitions remain the
same oracle transitions as the primary policy.

Seeds are:

- easy: `2026084400` through `2026084407`
- medium: `2026084408` through `2026084415`
- hard: `2026084416` through `2026084423`

Report each replicate and use the mean paired truth-cell loss over the eight
replicates for the frozen random-worse-than-d3 gate.
