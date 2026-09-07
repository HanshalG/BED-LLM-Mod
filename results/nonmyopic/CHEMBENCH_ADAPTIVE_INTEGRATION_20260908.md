# Adaptive integration: one-step qualification

New componentwise adaptive Gaussian integration retains the raw likelihood and
log-domain posterior. Subdivision starts at component peaks and widths; it is
not an observation discretization. It stops on evaluation/time caps, integration
nonconvergence, invalid continuation values or excess estimated error.

The caller supplies a global continuation bound. Omitted Gaussian tails have an
analytic contribution bound. QUADPACK's interior error is an estimate, not a
rigorous certificate. Recursive continuation error is not included automatically.

The new audit reuses the eight immutable synthetic reference integrals, not new
chemistry outcomes. It requires absolute error <=1e-6 and regret <=0.001, with
10,000 evaluations and 10 seconds per integral. All eight pass: maximum absolute
error 1.7643e-13, maximum evaluations 840. The earlier order-nine failed artifact
is unchanged. This is numerical-method development on opened synthetic fixtures,
not independent scientific validation or LLM efficacy evidence.

Raw integration, raw belief and horizon tests: 61/61. Tests include the previously
failed unequal-noise case, a close-action gap below0.001 with a known analytical
ordering, constant continuations, invalid values and resource exhaustion.

Next dependency: propagate continuation-error estimates through bounded h2/h3
optimization and compare against independent small multi-step references. This
one-step pass does not authorize the chemistry panel or paid calls. The integrator
is not yet wired into the horizon planner. No chemistry outcomes, model calls,
cluster activity or automation changes occurred.

Result: `chembench_adaptive_integration/20260908-v1/RESULT.json`, including source
hashes and per-case error/evaluation diagnostics.
