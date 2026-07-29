# Number Game Pooled Replication Synthesis-64 Result

Run: `number-game-pooled-replication-synthesis64-20260729T113442Z`

Status: **retrospective policy robustness positive**. This zero-call
cohort-stratified synthesis cannot rescue or reclassify either source, and
the prospective second-refresh mechanism null remains binding.

## Policy Versus Myopic

Across the two disjoint 32-tree pooled-Qwen cohorts:

- depth-three Brier: `0.103632`;
- myopic Brier: `0.116726`;
- relative Brier reduction: `11.22%`;
- stratified bootstrap difference:
  `[-0.0178237, -0.0085894]`;
- wins/ties/losses: `47/5/12`.

The source mean Brier differences are `-0.0110552` and `-0.0151328`. Their
cohort-one-minus-two contrast is `0.0040776`, with interval
`[-0.0051074, 0.0133960]`, so there is no evidence that the Brier effect
differs between cohorts. Both source policy-efficacy gate sets and every
frozen pooled robustness check pass.

Pooled Hamming improves by `6.55%`, but its interval
`[-0.0043739, 0.0010127]` crosses zero. The cohorts differ: the first is
slightly adverse and the second strongly positive. Pooled coverage improves
by `0.95` percentage points, also with an interval crossing zero; the source
coverage directions are opposite.

## Second-Refresh Support

Merged retained-plus-regenerated support versus parent-only:

- pooled relative Brier reduction: `4.36%`;
- stratified interval: `[-0.0083398, -0.0013075]`;
- roots differ on `41/64` trees (`22`, `19`);
- wins/ties/losses: `23/23/18`;
- source mean differences: `-0.0080022`, `-0.0014465`;
- cohort-one-minus-two contrast: `-0.0065557`, interval
  `[-0.0137129, 0.0004501]`.

The pooled interval is below zero, but it combines a retrospective positive
source with a prospective source that failed three of four frozen mechanism
gates. The between-cohort contrast also narrowly includes zero. The aggregate
therefore describes average support value but does not establish prospective
replication of the regeneration increment.

Merged support versus generated-only:

- pooled relative Brier reduction: `2.75%`;
- stratified interval: `[-0.0058129, -0.0003282]`;
- roots differ on `40/64` trees (`18`, `22`);
- wins/ties/losses: `24/24/16`;
- source mean differences: `-0.0026037`, `-0.0032669`;
- source-effect contrast interval: `[-0.0049449, 0.0058716]`.

The nearly identical directional source effects and pooled interval below
zero make retention the stable support component. Newly regenerated support
beyond parent-only remains the unconfirmed component.

## Interpretation

The robust result is the broad one: independently regenerated,
path-conditioned depth-three planning repeatedly selects better first
queries than myopic EIG on the exact canonical endpoint. The narrower causal
claim does not survive prospective confirmation. The evidence supports
retaining compatible hypotheses while refreshing support, but it does not
yet show that the refresh itself improves over retained parent support in a
fresh preregistered cohort.

Model calls: `0`. Cost: `$0`.

Public `RESULT.json` SHA-256:
`fb735fb557c3e24a6cb5c0d5fc518e497e01795a7d1273dce51cb83e63038a82`.
