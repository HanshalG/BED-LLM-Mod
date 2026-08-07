# RegretBench Branch-Draw Decision Protocol

Date frozen: 2026-08-07, before any RegretBench policy response or endpoint.

## Scope

Interpret only a complete artifact from the frozen RegretBench branch-draw
fidelity audit. This decision instrument is descriptive engineering guidance.
It cannot alter the source result, any mechanics or science gate, confirmation
authorization, claim tier, or paper conclusion.

The regions below are evaluated in order. Correlations are Spearman values.
`P+` denotes paired-bootstrap probability of positive correlation. `Pmean>both`
denotes paired-bootstrap probability that the two-draw mean correlation exceeds
both individual draw correlations.

## Ordered Decision Regions

### 1. Insufficient Information

Use `insufficient_changed_roots` if any condition holds:

- fewer than `16` changed dynamic-versus-refresh roots;
- fewer than `90%` of the requested paired bootstrap samples are retained; or
- any draw or ensemble correlation, interval, or positive-correlation
  probability is unavailable.

Next action: bank the diagnostic. Do not select a model, prompt, or draw count
from it. A new cohort is justified only by a separately powered protocol.

### 2. Actionable Draw Variance

Use `averaging_reduces_draw_noise` when all conditions hold:

- ensemble correlation is at least `.15` and ensemble `P+` is at least `.80`;
- `Pmean>both` is at least `.80`;
- ensemble RMSE is at most `90%` of the lower individual-draw RMSE; and
- draw prediction RMS gap is at least `.02` Brier.

Next action: do not change model or prompt first. On an untouched development
cohort, preregister a same-model four-draw instrument with the first two draws
identical to this design, compare two-draw and four-draw ranking fidelity, and
retain exact root/task common random numbers. Confirmation remains untouched.

### 3. Actionable Shared Error

Use `shared_ranking_error` when all conditions hold:

- draw prediction Pearson correlation is at least `.80`;
- draw prediction RMS gap is at most `.02` Brier;
- the maximum draw or ensemble correlation is below `.15`; and
- ensemble `P+` is below `.80`.

Next action: do not buy more same-model draws. Use completed development
histories only to diagnose transition prompt, support, likelihood, and model
bias; then freeze any repair on a new holdout cohort.

### 4. Actionable Draw Sensitivity

Use `draw_sensitive` when all conditions hold:

- absolute correlation difference between draws is at least `.30`;
- the stronger draw has `P+` at least `.80`; and
- the weaker draw has `P+` at most `.50`.

Next action: run a schema-clean four-draw smoke first. A later untouched cohort
may compare draw counts, but no individual seed or favorable draw may be
selected or discarded.

### 5. Fidelity Adequate

Use `fidelity_adequate_no_draw_escalation` when ensemble correlation is at
least `.15` and ensemble `P+` is at least `.80`, but no earlier region applies.

Next action: do not optimize draw count from this result. If the primary effect
is null, treat it as substantive under the current simulator rather than a
demonstrated ranking-fidelity failure. If the primary effect passes, proceed
only through its already frozen confirmation rule.

### 6. Inconclusive

Use `inconclusive_no_adaptive_repair` otherwise. Bank the result and do not
choose a model, prompt, favorable subset, or confirmation action from it.

## Reporting

The interpreter must emit the selected region, every literal boolean used by
the ordered classifier, exact source hashes, and the fixed next action. It uses
zero model calls and zero paid endpoints. Mechanics-failed fidelity artifacts
map to `unavailable_mechanics_failed` before these regions are evaluated.
