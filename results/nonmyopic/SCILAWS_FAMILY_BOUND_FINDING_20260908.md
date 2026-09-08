# Family-revelation bound exposes integration optimism

The previous turn cleared h2 runtime and exposed the h3 node limit. This turn
audits a mathematical lower bound before attempting branch pruning. No source
observations or hidden model labels were accessed: the label revelation is a
thought experiment inside the agent's generic reference mixture.

## Bound

For component k with current coefficient precision Lambda_k, target feature
matrix X_k, fixed target weights W and proposed action multiset S, define

    L_k(S) = E[sigma_k^2 | history,k] *
             (trace((Lambda_k + sum_{a in S} phi_ka phi_ka')^{-1}
                    X_k' W X_k) + noisy_target_indicator).

If the family were revealed, the expected posterior noise variance is a
martingale. Target leverage depends on the action multiset, not measured values.
By backwards induction, optimal future action choices under this known-family
model minimize that leverage independently of outcomes: remaining risk has a
common positive noise-variance factor. Thus enumerate action multisets exactly
for at most3 steps; permutations are unnecessary. The weighted sum over families
of their separate minima is a lower bound for the original continuous-model
Bayes risk because it grants additional information. A prescribed root is held
fixed and only its suffix is optimized. Unknown coefficients and noise remain
unknown even in this relaxation.

This proof concerns the stipulated homoscedastic conjugate mixture and its exact
predictive integrals, NOT actual SciLaws physics or a finite quadrature objective.
Floating-point calculations also are not interval-certified rigorous numerics.
The implementation is diagnostic only, with pruning authorization false.

## Finding

Artifact SCILAWS_FAMILY_BOUND_AUDIT_20260908.json, SHA256
e2d0c27ba1202f5f54e41f687d5ba531e042cd55de2aa0bc75bf8f4652affeec.
Inputs bind the exact geometry and completed terminal-batch preflight.

| Generic prior dimension | h2 family lower bound | Saved h2 estimate |
|---|---:|---:|
| 3 (baseball) | .165759530152 | .165661703681 |
| 2 (bird, spirometry) | .157273237273 | .157344171329 |
| 1 (remaining five tasks) | .149755892934 | .149849353378 |

Baseball's estimated h2 risk lies .000097826472 below the lower bound, about
.0591% of the estimate. This is incompatible with exact integration; it is not
a better-than-oracle policy. The sign matters even though the absolute error is
small. The bound would numerically rule out all8 forced roots against that
approximate incumbent, illustrating why directly wiring it into pruning is unsafe.
For other dimensions seven root bounds exceed the saved incumbent estimate, but
that is not certified pruning either. Same-dimensional rows are not independent
source evidence, and no equal-four-query deployed benefit has been measured.

## Next numerical dependency

Check predictive moment conservation and use exact conditional-moment identities
where possible. For the last observation, the integrated within-family variance
has an analytic expression; only the between-family term needs numerical
integration. This suggests a Rao-Blackwellized numerical reference, but it must
be integrated consistently with displayed tree risks. Changing only the fast
terminal hook while displaying old branch-summed risks would be inconsistent.

An alternative is a moment-preserving predictive quadrature with explicit
refinement checks. Neither correction is implemented or qualified here. First
require agreement with analytic martingale identities, independent mixture-density
integration and action-value refinement, then revisit search bounds and h3 cost.
Do not use the old optimistic incumbent to claim pruning or a depth win. The
prior, source physics, support and scientific thresholds stay fixed; numerical
correction is not permission to reopen an efficacy null (none was measured).

Seven new tests pass in.36s, including analytic known-family solutions at depths
0-3, forced-first-action behavior after conditioning, independent ordered-sequence
enumeration, and invalid horizons. These tests establish the bound calculation,
not integration accuracy or source calibration. Scoped lint passes.
No paid calls, source measurements or running processes. Account/ledger unchanged,
automation paused. Full research goal remains unfinished.
