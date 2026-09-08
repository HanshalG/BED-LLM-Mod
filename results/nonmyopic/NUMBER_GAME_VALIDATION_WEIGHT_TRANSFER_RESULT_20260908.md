# Validation-only weighting transfers, with a small first-step gain

Retrospective descriptive analysis frozen at pushed `32c5a226` before its run.
All 32 trees, 256 roots and 5,744 saved target-query cases included. Eight separately
seeded validation-rule draws per tree fit ONE scalar for that tree, with equal draw
weights. All 32 weights were saved before the target-field pass. No model calls,
new endpoint generation, inference cost, prior-result changes or gate authority.

## Results

| Predictor after first observation | Mean target Brier |
|---|---:|
| Initial-support filtering | 0.1875159620 |
| Original uniform retained refresh | 0.1950510101 |
| Validation-weighted interpolation | 0.1860525022 |

The validation-weighted predictor improves **0.7804%** over initial filtering,
winning on 31 trees and losing on one. Paired mean difference is -0.00146346,
SD 0.00146224. Against uniform refresh it improves **4.6134%**, winning on all
32 trees; paired mean difference -0.00899851, SD 0.00347257.

Fitted refresh-step weights range 0.08243-0.63095, mean 0.29720, median 0.27663.
All eight-draw validation sets have a beneficial first-order direction. Target
direction was beneficial on all 32 trees in the separately banked diagnostic.
No root-specific or target-fitted weight is used; there is no sweep over priors,
loss functions, calibration subsets or alternative formulas.

These SDs summarize paired tree differences; this is not a fresh significance
test or a prospective efficacy pass. The previously reported shared-proposal pool
has target mean 0.18617809, numerically close to 0.18605250. This analysis does not
establish a meaningful advantage over that larger-compute control.

## Interpretation

The overshoot diagnosis has a concrete transfer check: independently seeded
synthetic prior draws select substantially less than the original full update,
and those weights improve prediction on separate target entries. This supports
testing predictive weighting alongside executable proposal generation instead of
assuming uniform model mass or discarding the proposer.

The effect over the strong initial-filtering reference is SMALL. Restoring an
overaggressive update is not the same as achieving a large model-discovery gain.
Do not turn the 4.61% uniform-refresh comparison into an implied 4.61% advantage
over the strongest baseline, or describe this as a planning-horizon improvement.

## Data separation and limits

The production first pass materializes initial/first-step supports, root IDs,
validation draws and seed metadata only. It validates all eight seeds per tree
against the frozen schedule and rejects overlap with that tree's target seed.
The second pass materializes target rules only after the complete weights file is
written. Target concepts can still overlap validation concepts naturally; no
outcome-dependent deduplication/exclusion is performed. The exact parent initial
and uniform-refresh losses replay at every root.

Crucially, the analyst had already examined aggregate TARGET results before this
new diagnostic was designed. Computational exclusion of target fields from fitting
does not undo that adaptive analysis. This is target-excluded fitting on an old
bank, NOT untouched held-out confirmation, independent physical calibration, or
evidence for transfer to a new model/interface/environment.

Validation draws come from the same historical target-model family and prompt;
they are generated rules rather than new physical observations. The fitted scalar
is a Brier-calibrated mixture coefficient, not a Bayesian mechanism probability.
This offline fit also does not empirically test `core/prequential.py`, whose weights
learn from subsequent query outcomes. Those two weighting methods must be named
and evaluated separately, not conflated.

## Verification and next action

Six calibration/direction tests pass (0.38s), including the exact convex solution,
boundary cases, equal draw rather than flattened-example weights, target-field
nonuse in fitting and seed-overlap rejection. Scoped lint and bound-source checks
pass. All parent baseline losses replay; saved aggregate arithmetic checks pass.

The next experiment should test the full proposer-plus-weighting mechanism on
fresh, prospectively fixed tasks with initial, uniform-refresh, calibrated-refresh,
history-blind/shuffled and symbolic controls, then measure actual non-myopic value.
This result is evidence for that design choice, not permission to rerun a closed
Number Game interface, alter an old threshold, or spend without a dependency-valid
new protocol and budget authorization.

The full goal remains incomplete. No cluster use; automation remains paused.
WEIGHTS SHA256: `77bc80bebd4561cb3456ae11b38ef937fdb9f41c2bf7d0da23b6d139246cd732`.
RESULT SHA256: `a7d47ae0269dc950d8d08e23fa099039d8e4c0d91d55e3277317e264b3c8c16f`.
