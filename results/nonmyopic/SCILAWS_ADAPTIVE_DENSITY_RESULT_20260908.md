# Density acceleration preserves reference, not sufficient for h2

The terminal cProfile diagnostic attributed most time to repeated scipy.stats
logpdf and scipy.special logsumexp setup. Precompute Student-t normalizing and
scale constants, evaluate the same log density using a stable hypot expression,
and reduce vector log masses with numpy.logaddexp.reduce. No quadrature node,
domain, tolerance, hypothesis, action or resource-cap changes.

Ten focused tests pass in.87s; scoped E4/E7/E9/F lint passes. Density comparisons
against scipy.stats.t include seven central/tail observations up to +/-1e150,
three degrees of freedom and three scales; independent full-posterior integral
comparison still passes. Illustrative profiled terminal call falls from .030s
and40368 function calls to .003s and1691 calls, both150 integrand evaluations.
These single profiles are attribution evidence, not a general speed benchmark.

Code pushed99a28a2a before one unchanged adaptive-reference panel. Artifact
SCILAWS_ADAPTIVE_DENSITY_AUDIT_20260908.json SHA256:
85a0509322e32889f01f322e4844ca42ffeac8a557af739205c5fd6c82c2ac5b.

- All4 h1 references complete with the same240-420 evaluation counts.
- Maximum saved h1 risk difference from preceding independent reference is
  5.551115123125783e-17; three cases are bit-identical.
- All4 h2 references remain incomplete, now hitting100001 evaluations rather
  than five seconds. No partially completed root values become a valid reference.

This removes per-evaluation overhead but does not solve nested integration.
The scientific instrument is still unqualified for the planned deeper mixture
search, and the LLM-native endpoint remains untouched.

Next inspect conditioning of infinite-domain integrals under far-out synthetic
observations: conditional means/scales move, while the current adaptive domain
is always centered at zero in raw observation units. A deterministic affine
change of variable using predictive center/scale and its exact Jacobian can
be independently tested without changing distributions, tolerances or caps.
It is a numerical candidate, not a promised fix or deployment permission.
Preserve failed runs and do not rerun unchanged or raise their evaluation limit.

No source measurements or model calls, $0. Authenticated balance24.623306006 and
Sept8 London ledger unchanged. Process exited, automation paused, full goal
unfinished. Useful proposals, calibrated source inference, paired depth/control
endpoints and anticipated discovery remain to be established.
