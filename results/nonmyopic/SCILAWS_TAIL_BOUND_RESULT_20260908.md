# Analytic tail envelope established on the diagnostic panel

Frozen07c05a3a, one analytic sweep, no inner integration or source calls.
Five focused tests pass in.94s; scoped E4/E7/E9/F lint passes. Independent
quadrature verifies asymmetric Student-t tail moments and the full conditioned
target second moment; widening the interval reduces the bound.

Artifact SCILAWS_TAIL_BOUND_AUDIT_20260908.json SHA256:
57ff42031bf9e2d26d111a0055053036fead16092217a96e4cec3860d30af785.

At the fixed1e-6 tail-risk allowance, selected radii in predictive standard
deviations and bounds are:

| History | Action0 radius / bound | Action1 radius / bound |
| --- | --- | --- |
| Empty | 64 / 4.387e-7 | 64 / 1.133e-7 |
| Positive | 32 / 2.709e-7 | 32 / 7.231e-7 |
| Negative | 32 / 9.584e-7 | 32 / 2.137e-7 |
| Contradictory | 32 / 3.837e-8 | 32 / 2.515e-8 |

Thus all8 admit a finite interpolation interval with an explicitly bounded
omitted contribution. A narrow cutoff would be invalid: empty-history action0
still has bound .001799 at radius8 and .00011254 at radius16. Tail probability
alone would underestimate the relevance of large squared losses.

The bound follows from a feasible zero target prediction and uses unnormalized
tail moments, not truncated/renormalized likelihoods. For a future interior
estimate, the tail is[0,B]; it cannot silently be replaced by zero. This is an
analytic continuous-working-model bound, subject to floating-point calculation,
not a statement about source misspecification or arbitrary approximate updates.

## Next interpolation design

The wide intervals favor deterministic asinh-spaced nodes, concentrating points
near predictive mass while covering both tails. Approximate each individual
continuation action value before minimizing, and normalize its quadratic growth
by1+z^2, where z is the standardized first observation. Since E[1+z^2]=2,
a genuine uniform normalized error epsilon would imply integrated error<=2epsilon;
sampled held-out checks alone must not be mislabeled as a uniform proof.

Freeze node counts/check points, interpolation rule, error gate and budget before
the new diagnostic. Compare against direct adaptive inner integrals at disjoint
check points, retain both actions and all histories, and count training/check
integrals against the same per-plan cap. Keep explicit tail and inner error
contributions. No source/LLM stage follows automatically from interpolation
passing sampled checks. The full sequential planning and scientific gates remain.

No source observations/modelcalls, $0. Account/Sept8 ledger unchanged, no process
running, automation paused, full research goal unfinished.
