# Matched-budget expected gain is small

The frozen public-prior diagnostic completes in 24.051 seconds, without a hidden
world constructor, new source endpoint, model request or paid call. It evaluates
the DEPLOYED myopic policy after all three measurements, not its one-step root
score. Every subsequent action uses the current posterior and full remaining
menu. Independent scalar-policy enumeration tests agree with the batched evaluator.

| Quadrature Order | Myopic Three-Measurement Risk | Optimal Three-Measurement Risk | Relative Gain |
|---|---:|---:|---:|
| 32 | 0.005879624 | 0.005850932 | 0.4880% |
| 64 | 0.005873013 | 0.005843404 | 0.5041% |

The 64-branch optimum is reused from the verified source pilot, not recomputed.
Its action values and the myopic root reproduce the banked policy. The 32/64
value differences are 6.61e-6 and 7.53e-6; the difference between the two estimated
gains is 9.16e-7. These are refinement diagnostics, not rigorous integration-error
certificates. Both orders meet the unchanged 0.001 absolute refinement criterion.

The 46.1% observed eight-world improvement is therefore NOT a similarly large
gain predicted by this finite model when measurement budgets are matched. This
does not prove that the empirical estimate is wrong: the source worlds were
drawn from continuous parameter boxes, whereas this calculation averages over
the 16-particle accessible prior. Model approximation, finite-sample variation
and the round-indexed noise coupling all remain relevant. It does show that this
instrument predicts only small non-myopic headroom on its current initial state.

The old h1 root score (~0.0220) and h3 root score (~0.00584) were values at
DIFFERENT optimized horizons. Treating their difference as a three-measurement
policy benefit would be incorrect. The appropriate h1 comparator is ~0.00587.
h2 and h3 remain structurally identical after their common first action under
the three-measurement budget, so extra trials cannot establish a strict h3 gain.

The original engineering_pass and all primary results remain untouched. Do not
use that screen alone to authorize LLM calls or a headline on this formulation.

## One bounded conditional-opportunity test

Before giving up on source chemistry, freeze exactly one natural extension:
a common uninhibited high-substrate calibration at [10,0,10,0,1,310,7], followed
by the original four-assay menu and three new measurements. Four fixed quartiles
of the public calibration predictive distribution define four contexts. Evaluate
every context at orders 32 and 64; select none by outcomes. This is a coarse
public conditional-opportunity test, not a new source efficacy experiment.

The full-budget value of receding h2 equals the h3 action value at h2's chosen
root, because its remaining two-step continuation is optimal. A separate
independent scalar-execution test checks that identity. The diagnostic requires
at least 5% aggregate improvement at both adjacent depths and both quadrature
orders, plus the existing refinement tolerance. These new prospective criteria
do not retroactively change the original pilot or the present descriptive audit.

Protocol: `CHEMBENCH_CALIBRATED_CONTEXT_OPPORTUNITY_PROTOCOL_20260908.json`.
A null closes this exact conditional proposal: no calibration-design, quantile,
noise, target or seed sweep to rescue it. No paid call or LLM efficacy claim is
authorized. The useful-proposal gate and broader LLM-native objective remain
unfinished regardless of this numerical test.
