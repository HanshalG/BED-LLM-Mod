# Adaptive per-action fit, separate from failed fixed grid

Freeze before one four-history/shared-budget diagnostic. Use17 uniform asinh
nodes initially; cubic splines model smooth individual action values rather than
PCHIP's shape-preserving derivative rule. Midpoint discrepancies>2e-5 cause local
bisection, all failed intervals in a pass, at most8 passes and65 fitting nodes.
Cache exact evaluated points within each root; every new direct inner solve is
charged to the same five-second/100000 per-plan budget. No history/target change.

After midpoint convergence, evaluate two golden-ratio interior points in every
final interval, disjoint from every fitting AND adaptation point. These final
checks may not be used to refit. Max normalized error remains2e-5, inner error
estimate limit1e-7, tail allowance1e-6. A cap or fresh-check failure emits no fit
score. Reject negative/nonfinite interpolated values instead of clipping them.
No extrapolation. Keep full density/Jacobian and explicit[0,B] tail contribution.

Adversarial tests must reject a function whose oscillation is invisible at both
initial knots and refinement midpoints but visible at fresh checks. Even passing
sample checks is not a uniform error certificate, so source/depth3 authorization
remains false. The failed fixed65-node PCHIP route stays unchanged and closed.
