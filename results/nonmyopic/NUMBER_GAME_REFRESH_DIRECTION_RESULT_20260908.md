# Refresh direction is useful; the uniform update overshoots

Retrospective descriptive analysis frozen at pushed `1948221a` before executing
the decomposition. All 32 trees and 256 roots from the complete parent diagnostic
are included, with its unchanged target/root/tree weights and fixed 101-number
target domain. No new model calls, endpoints, inference cost or paid authority.

## Result

For initial-filtered prediction p0, retained-refresh prediction p1, and d=p1-p0,
the loss change along the common scalar path p(t)=p0+t*d is exactly:

    delta Brier(t) = t*A + t^2*B
    A = -0.009380078974273464
    B = +0.016915127079411996
    delta Brier(1) = +0.007535048105138532

The cross term A is NEGATIVE on every one of the 32 trees. Thus sufficiently small
positive movement in the observed update direction improves each tree's average
held-out loss. The full original step is too large on 30 trees; it improves on two.
Every root's exact A+B equals the parent's saved refreshed-minus-initial loss.

This sharpens, rather than contradicts, the preceding finding that uniform refresh
worsens loss by 4.02%. The candidates contain useful predictive information under
this directional test, but the original weighting gives their aggregate prediction
too much influence. Both statements concern this saved predictor and target bank.

## What we did not do

No optimal t was computed or selected, no calibrated predictor was evaluated, and
no improvement is claimed for a deployable weighting method. Target-informed
interpolation would be retrospective tuning, not held-out efficacy. The negative
derivative is a diagnostic of prediction geometry, not Bayesian posterior calibration
or evidence that a fixed smaller weight transfers to fresh tasks.

The result does not establish a universally good proposer, identify which individual
new rules are useful, or eliminate target-prior mismatch. It also does not prove
usefulness after later observations or monotonic gains with planning horizon.

## Architectural consequence

Do not discard the proposer solely because its uniform pooled update loses. The
next new proposal/update study should separate structure generation from assigning
predictive mass. In particular, adding ten near-equivalent valid models should not
automatically give an explanation ten times the mass.

Use a prospectively specified complexity/family prior or calibrated predictive
mixture, with calibration learned on disjoint histories and tested on fresh tasks.
For prequential calibration, score a proposal only on observations arriving AFTER
that proposal was made. Refitting on the same observation that elicited a new
structure is not independent evidence of its reliability. Preserve the original
model group as an explicit reference and log its mass, new-group mass, predictive
changes and subsequent out-of-sample loss.

Any real-history weighting or acceptance rule must use only information available
at that time. The current target bank cannot choose its coefficients, priors or
gates. A future paired comparison needs initial-filtering, original uniform refresh,
calibrated refresh, history-blind/shuffled and actual symbolic proposal controls,
with proposal/inference costs stated. This is a design consequence, not a frozen
new paid experiment or permission to rerun a terminal Number Game interface.

## Verification and disposition

Three focused tests pass (0.16s), testing wrong-direction versus overshoot examples,
the exact quadratic identity at multiple steps, and length rejection; lint clean.
Bank/compiler/parent bindings and every root's exact parent loss replay passed.
Raw components and all roots are banked in NUMBER_GAME_REFRESH_DIRECTION_20260908.json.

Keep earlier nulls and thresholds unchanged. The full objective is still incomplete,
but the next action is now better targeted: validate a proposer-plus-weighting
mechanism before further environment construction or deeper planning. No cluster
use, no account balance asserted, and automation remains paused.
