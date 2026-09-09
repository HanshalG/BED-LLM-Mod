# Adaptive parameter integration development result

V6, frozen code 4af4a8e7, resolves both opened numerical fixtures within the
unchanged 400000 likelihood-row cap. No model calls, paid endpoints, source
simulation or changes to old scientific gates.

| Fixture | Likelihood rows | Absolute log-evidence error | Maximum mean error | Maximum relative variance error |
|---|---:|---:|---:|---:|
| One parameter | 1640 | 5.06e-14 | 7.77e-15 | 2.13e-12 |
| Four modes | 313022 | 7.11e-15 | 1.78e-15 | 2.11e-15 |

References are the independent 2048-point Gauss-Legendre integrals from the
previous audit. All numbers are development-fixture agreement, not a prospective
held-out qualification or proof of arbitrary posterior coverage.

## What changed

The implementation uses [SciPy cubature](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.cubature.html),
which adaptively subdivides an integration box and returns componentwise estimated
errors. It is not guaranteed to converge for every integrand.

An analytic Gaussian likelihood ceiling provides stable scaling. Evidence is
integrated with relative tolerance and zero absolute tolerance. Its discovered
regions are reused for predictive moments, computed using both 21- and 15-node
Gauss-Kronrod rules. Predictions are centered at a numerically sampled high-
likelihood point to avoid variance cancellation; that point uses no true parameter
or target observation. Evidence, estimated normalized moment errors and agreement
between rules must all pass the original outer thresholds. One additional
prediction-only row selects the center and is recorded separately from likelihood
evaluations. Only one or two independent transformed parameters are supported.

## All iterations retained

- V1: four-mode agreement; one-parameter evidence rejected after the absolute
  tolerance stopped at an almost-zero integral without finding the narrow peak.
- V2: a relative-error evidence pilot found the peak, but restarting moment
  integration globally lost that partition information; one parameter unresolved.
- V3: reuse regions fixed one parameter; splitting all regions exceeded the cap
  for four modes.
- V4: two rule comparison reduced work but four modes still reached the cap.
- V5: inner precision matched more closely to the unchanged outer criteria;
  four modes passed, while conservative errors on uncentered variance rejected
  one parameter.
- V6: centered moments resolve that cancellation without changing the outer
  acceptance thresholds or work cap; both fixtures agree with references.

All six artifacts remain saved under ADAPTIVE_PARAMETER_REFERENCE*. These were
transparent numerical development iterations on already-opened fixtures, not six
attempts on a sealed scientific cohort. Do not present V6 as a held-out success or
erase failures. Library estimated errors and cross-rule agreement can still miss
an unsampled narrow mode. The result does not qualify higher-dimensional laws.

## Research consequence

Stop the generic sampler sweep here. Next connect this moment/evidence backend to
the existing safe executable-law interface, preserving full-history replay and
equal prior mass for canonical laws. Test multiple competing structures against an
independent known mixture before any paid semantic-proposer call. Do not silently
discard unintegrable structures or restrict a previously frozen scientific protocol
to two parameters: unresolved mixtures fail closed, and any lower-dimensional
proposer interface requires a new prospective study.

The backend returns conditional evidence/moments, not a particle state or a
simulation-ready generative posterior. That distinction must remain explicit in
the bridge. Later planning still requires joint predictive distributions and
validated branch updates, useful LLM structural discovery, ordinary horizon
headroom and paired compute-matched controls. None is established by these tests.

Authenticated balance remains $23.693468061; conservative daily remaining allowance
$4.11174654. Previous turn was progress, and this turn implements and validates a
numerical dependency. Full goal active and unachieved.
