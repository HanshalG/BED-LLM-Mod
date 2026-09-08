# Horizon-aware variance correction

Previous turn found a deep error in the current-variance control variate. This
turn adds HorizonControlVariateMixture as an explicit alternative, leaving the
old estimator and failed diagnostics banked. A depth-aware planner hook takes
precedence over the existing depth-independent hook. Terminal risk still uses
the already verified analytic one-step correction. Saved tree offsets expose
the numerical correction at every nonterminal node.

## Potential and implementation

Use U_d(s), the weighted family-revelation optimum with d observations remaining,
as the control variate. Given current action a, each family's updated precision
is fixed before seeing its outcome. Its optimal remaining action-multiset
coefficient is therefore fixed, and E[U_(d-1)(s_next)] is computed analytically
using expected noise variance. The chance operator is

    T_d V(s,a) = E[U_(d-1)(s_next)] + Q[V(s_next)-U_(d-1)(s_next)].

It leaves all family probabilities, raw likelihoods, observation nodes and the
physical budget unchanged. Coefficients use the full action menu with repeats
as a relaxation, cached in1024 bounded entries; noise-moment differences have a
256-entry cache. The implementation only accepts depths1-3. When repeats are
forbidden the full-menu potential is still a valid control variate, but the
single-family exactness fixture below is specifically the repeat-allowed case.

## Verification and bounded audit

184 tests pass in19.22s across SciLaws and the ordinary/raw/batch horizon suites;
scoped lint passes. New checks cover analytic single-family risks at h1/h2/h3
in adaptive and open-loop modes, explicit tree reconstruction and a mixed-family
correction independently computed by scalar family-bound enumeration.

Implementation/audit selection pushed atfb7fbbc5 before execution. Same cases,
thresholds, five-second cap and100000-node cap as the preceding deep audit.
Artifact SHA256:
680312a3ad5190ef8cb3f5777f77ae95b17f6a9b9e0e9dd4f06bccad68f23674.

- All9 analytic cases (orders4/8/16 x depths1/2/3) complete and pass unchanged
  1e-4 error tolerance. Maximum error is2.7755575615628914e-17.
- Public h2 completes7/8 cases in1.617-2.934 seconds; all completed cases pass
  the necessary family-bound consistency check.
- Baseball h2 hits the five-second limit. It has no completed value and must
  not be counted as a bound pass. Do not infer its cause from elapsed time alone.
- No full public h3 rerun, source observations, model calls or scientific endpoint.

This establishes the analytic-fixture fix, not exact mixed-family integration
or a full runtime pass. Same-dimensional public priors remain identical; the
seven completions are not seven independent positive scientific results.

## Search implication

The new operator has a useful lower-bound property for its OWN numerical
objective, subject to floating-point verification. At depth0 the target risk
V_0 is U_0 plus nonnegative between-family variance. If V_(d-1)>=U_(d-1), then
nonnegative quadrature masses imply

    T_d V(s,a) >= E[U_(d-1)(s_next)] >= U_d(s).

The first right-hand expression is the forced-root family-revelation bound.
Minimizing actions preserves the induction. This property did not hold for the
old optimistic uncorrected chance operator. It permits investigating equivalent
branch-and-bound search of this corrected finite estimator without assuming its
integrals equal the true continuous risk. It is not a source-world guarantee.

Next implement optional bound-ordered search with conservative numerical margins,
retain explicit lower bounds for pruned roots (never report them as exact action
values), and compare selected policies/values to complete enumeration on tractable
mixed-family cases. Do not silently skip an action without recording why its bound
exceeds a feasible incumbent. Then test the full unchanged capped panel. Deeper
mixed-family integration refinement remains a separate requirement before source
execution; pruning equivalence cannot substitute for it.

Account and current London ledger unchanged at zero spend, no active process,
automation paused. Full paired protocol, source usage obligations, useful LLM
proposals/refresh and anticipated-discovery evidence remain unfinished.
