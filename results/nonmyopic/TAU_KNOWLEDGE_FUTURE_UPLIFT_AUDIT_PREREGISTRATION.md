# tau-Knowledge Two-Split Future-Uplift Audit Preregistration

## Status

Frozen before computing any uplift score or endpoint metric. This is a
zero-call, post hoc audit on two disjoint, already-open 20-task tau-Knowledge
confirmations. It is a mechanism analysis, not a new held-out result.

## Motivation

The deployed first-link comparison asks whether a full-tree scorer ranks total
two-step root value better than an isolated first-result scorer. A stronger
mechanism question is whether the *change* in score caused by exposing the
future tree ranks the exact incremental value supplied by that future.

## Frozen Artifacts

- First-link V2 confirmation SHA-256:
  `dfbf597f8405b109d61d90206606962b5fbda439c8b81f086a9592c07fa247d1`.
- Receding V3.1 confirmation SHA-256:
  `f8b120b0d2b98f9f10eb0ef9251262a6a41b20d4d90d724ee4926c141ec275ae`.
- The two 20-task ID sets must be disjoint.

Both artifacts contain identical per-task interfaces: five roots, isolated
myopic scores, full-tree scores, exact first-result documents, four follow-up
result sets per root, and hidden required-document IDs.

## Frozen Estimand

For root `r`:

- `immediate(r)` is distinct required-document coverage in its first results;
- `total(r)` is maximum distinct required-document coverage over its four
  first-plus-follow-up pairs;
- `future_gain(r) = total(r) - immediate(r)`; and
- `score_uplift(r) = full_tree_score(r) - myopic_score(r)`.

Within each task, use pairwise ranking accuracy of `score_uplift` against
`future_gain`, giving half credit for a tied score on endpoint-distinct roots.
Score offsets cancel within task. Roots with equal future gain are not
comparable.

## Frozen Metrics

For each split and pooled across 40 tasks, report:

- comparable root-pair count;
- uplift pairwise accuracy;
- task-level exact one-sided sign-flip test above chance;
- count of tasks with at least one comparable pair;
- full-tree-score and myopic-score accuracy against future gain;
- score-uplift-selected future-gain total;
- myopic-selected and full-tree-selected future-gain totals; and
- score-uplift root agreement with deployed full-tree selection.

Also report descriptive calibration links:

- myopic score versus immediate coverage; and
- full-tree score versus total two-step coverage.

## Frozen Interpretation

- **Strong future-value signal:** each split has at least 30 comparable pairs
  and uplift accuracy above `.50`; pooled uplift accuracy is at least `.60`;
  and pooled task-level one-sided `p <= .05`.
- **Directional future-value signal:** both splits have uplift accuracy at
  least `.50` and pooled accuracy is at least `.55`, but a strong condition
  fails.
- **Null/adverse:** otherwise.

The audit may establish that future-tree exposure changes semantic scores in a
direction consistent with exact incremental retrieval value. It cannot prove
that regenerated belief text, rather than future queries/documents, mediates
that change.

No alternate score transform, coefficient, normalization, subset, or
threshold follows the result. OpenRouter and OatML use are both zero.
