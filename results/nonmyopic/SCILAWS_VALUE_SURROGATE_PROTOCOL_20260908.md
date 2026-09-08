# Frozen per-action value interpolation diagnostic

All four opened synthetic histories, two roots, one shared five-second/100000
evaluation budget per history. Preserve Student-t models, all actions and targets.
Select tail radius from4/8/16/32/64 using analytic bound<=1e-6. Fit each individual
continuation action's value divided by1+z^2 at65 uniform asinh(z) nodes. Use
PCHIP with extrapolation disabled. Validate at all64 disjoint interval midpoints
using direct adaptive terminal integrals; never train on those checks.

Normalized midpoint max error must be<=2e-5 and encountered normalized inner
error estimates<=1e-7. Otherwise return validation_failed without a planning
score. On pass integrate the actual minimum of the separate interpolants with
full density/Jacobian, epsabs=epsrel=1e-8 and100 subdivisions; retain analytic tail
contribution[0,B], not renormalized mass. Count all training, checks, integration
and bound attempts against the shared cap. Both roots must complete.

Sample checks do not prove a uniform bound; tail_only_interval is NOT a full
confidence or numerical-error interval. No source/deployment authorization even
on a pass. Compare completed candidates to independent saved reference evidence
before considering further qualification. Freeze before one execution; no node
count/threshold/history adjustment to rescue the observed result.
