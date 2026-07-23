# Animals Branch-Support Ranker Development Plan

Status: **development only; inspected targets**.

## Hypothesis

The size-only belief-recall ranker could not see which hypotheses each
counterfactual branch actually recovered. The closed-support Dynamic-Brier
score could not value an omitted truth at all. A target-blind ranker that sees
the actual regenerated Yes/No supports can assess semantic coverage of newly
generated hypotheses while remaining ignorant of the benchmark target.

## Frozen Development Probe

- Seed `24279`.
- Twenty states using the first 20 now-inspected capacity-holdout targets.
- Three ordinary production candidates per state.
- Unchanged non-thinking Gemma 4 26B belief generator and ranker.
- Ranker payload allowlist: history, current support, candidate text,
  predictive Yes/No probabilities, and actual regenerated Yes/No supports.
- Excluded: target, truth-coverage fields, immediate EIG, and all selected
  endpoint values.

Proceed to a separately frozen fresh holdout only if all 20 states complete,
the target-free payload audit passes, candidate-level Spearman association
with expected truth coverage is positive and exceeds immediate EIG, selected
mean expected truth coverage exceeds EIG, and wins exceed losses.

Failure stops this exact scorer without prompt repair, score blending, or a
fresh-target run.
