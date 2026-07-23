# Animals Multi-Sample Branch Ranker Development Plan

Status: **development only; inspected seed-24279 states**.

The four-list generator raised counterfactual branch-union truth recovery to
`8/20`, making branch-content ranking identifiable often enough for a second
development gate.

The scorer is the exact target-blind branch-support prompt committed at
`65f83fb`, with no wording or model change. It sees history, current support,
candidate questions, predictive branch probabilities, and the broader
regenerated Yes/No supports. Baseline immediate EIG is deterministically joined
from the immutable seed-24279 source for post-response comparison only and is
excluded from every model payload.

Proceed to a fresh sealed holdout only if:

1. all 20 responses parse with zero reasoning;
2. every payload rebuilds from the explicit allowlist and excludes target,
   truth-derived fields, and immediate EIG;
3. candidate-level Spearman with expected truth coverage is positive and
   exceeds immediate EIG;
4. selected expected truth coverage exceeds immediate EIG;
5. wins exceed losses.

Failure stops this scorer/interface combination without prompt or coefficient
repair.
