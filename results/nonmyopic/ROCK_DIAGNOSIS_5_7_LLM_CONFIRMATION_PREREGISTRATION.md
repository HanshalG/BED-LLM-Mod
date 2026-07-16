# Rock Diagnosis `5-7` LLM Depth Confirmation Preregistration

Registered: 2026-07-15, after one completed `5-7` exploratory pilot and before this
confirmation's first provider request.

## Claim

On the independently specified Figure 4 `5-7` Rock Diagnosis map, a two-step exact
incremental-EIG policy selected from LLM-proposed K=3 legal candidate cells has lower
final exact posterior entropy than both a shared-cell one-step policy and a
candidate-call-matched one-step width policy.

## Frozen Run

- Fresh seed `9175`; 30 paired trajectories; 8 rounds; K=3; same paper map/start,
  32-state exact posterior, full-vector MAP decode, CRN, transition, likelihood, and
  strict one-validation-feedback-retry interface as the `5-7` pilot.
- Same non-thinking `google/gemma-4-26b-a4b-it`, temperature `0`, 128 output tokens,
  exact prompt template, and three arms: `d1_shared`, `d2`, and
  `d1_call_matched_width`.
- Frozen committed runner/analyzer:
  `scripts/nonmyopic_rock_diagnosis_pilot.py --confirmatory`; paired 10,000-resample
  percentile bootstrap intervals; raw candidate requests, retries, and per-step traces
  retained. No action padding or programmatic replacement is permitted.

## Power And Cost

The eight-pair pilot found final-entropy reductions `0.6056 +/- 0.1503` SD versus
shared d1 and `0.5636 +/- 0.1603` versus matched width (standardized effects 4.03 and
3.52). The normal approximation is implausibly optimistic at this small n (one pair
for 90% power); the fixed 30-pair confirmation instead matches the powered `3-6`
confirmation and protects against winner's curse and non-normality.

The pilot cost `$0.06182011` for 8 trajectories, projecting `$0.23183` for 30. The
confirmation projects `$0.30` and is hard-capped at `$0.45` by the adapter. Together
with the exact gate and pilot this remains within the authorized total budget.

## Decision

Primary metric: paired `H(control)-H(d2)` final exact posterior-entropy reduction.
The confirmation passes only if both paired 95% bootstrap lower bounds are strictly
positive and all selected actions are legal, roots are shared, width's logical-call
allocation matches the d2 virtual root tree, and no cell exhausts its one feedback
retry. MAP accuracy, entropy AUC, truth log posterior, root actions, candidate
diversity, retries, and cost are secondary and reported regardless of outcome.
