# Public-prior kernel acceleration

## Scope and outcome

Zero model calls and zero hidden-world construction. The frozen physics SHA is
`8e1fc9df41d177fa80b2e500c22c663e3a78a980ecedaa355c667116cc2e1d36`.
All 16 prior particles, 64 quadrature branches, complete contingent action search,
fixed targets and the 60-second decision cap are unchanged.

The original public-prior h2 cProfile run took 1.366 seconds, including 1.071
seconds in mixture CDF evaluation. This identified inverse-CDF bisection as the
largest bottleneck, not terminal risk. These baseline timings were captured in
the task output, not a separate immutable JSON artifact.

Added bracket-verified Newton acceleration for ordinary probabilities, initialized
by a separated-mixture approximation used only as a guess. Full-mixture CDFs
validate the final narrow bracket. Unresolved roots and extreme tails keep the
64-bisection fallback; nodes whose brackets reach adjacent floating-point values
stop redundant evaluations. No likelihood component is removed or pruned.

The saved updated h2 profile completes in 0.577827 seconds at the same 49,408
processed states. Its root values agree with the original to about 1e-16 and its
selected action remains 1. This is a local profiled timing comparison, not a
general speedup guarantee. Indeed, the two-particle synthetic h3 qualification
took 10.21 seconds at 64 branches versus 3.94 seconds in the earlier V3 run:
the extra accelerator machinery can cost more than it saves for small mixtures.

The instrumented public-prior h3 run still reaches the unchanged cap at 60.011
seconds. Its 695 branch-batch calls spend 58.39 seconds inside branch generation,
including 20.04 seconds in selected CDF evaluation and 14.68 seconds constructing
44,354 envelope rules. Profiling adds overhead; this is not an uninstrumented
complete-pilot measurement. It establishes no efficacy result or completed h3
plan. No full pilot was relaunched in this checkpoint.

## Verification

- 99 focused horizon, raw-likelihood, integration, batch, envelope and pilot tests
  pass in 34.25 seconds.
- Four added independent 64-bisection comparisons cover 16-particle mixtures,
  equal/unequal noise, diffuse priors and concentrated posteriors. Existing
  zero-support, tiny-tail and scalar/batched full-policy tests remain passing.
- Envelope refinement V4 passes every unchanged one-step, three-step, refinement
  and constructed-adaptivity gate. The preflight now binds these exact kernel
  hashes and passes on the public prior.
- Scoped lint and whitespace checks pass. No protocol thresholds were relaxed.

## Artifacts and next dependency

- `chembench_public_kernel_profile/20260908-h2-v1/RESULT.json`
- `chembench_public_kernel_profile/20260908-v1/RESULT.json`
- `chembench_envelope_refinement/20260908-v4/RESULT.json`
- `chembench_horizon_pilot/preflight-20260908-v4/RESULT.json`

The next dependency remains numerical throughput under the frozen source caps,
not more experiments or model spending. Target the per-belief root/CDF and
envelope loops with a separately tested compiled or vectorized implementation;
retain independent bisection equivalence, all hypotheses and fixed integration
nodes. Avoid claiming a universal gain from the h2 timing. Qualify any subsequent
kernel revision in a fresh artifact before a new complete-pilot attempt.

The eight-world source panel, its paired controls and replay validation, and the
later useful-LLM-proposal gate remain unfinished. There is still no source-panel
evidence of monotonic depth gains and no new LLM efficacy evidence. Automation
remains paused; no cluster, cleanup, paid request or old endpoint reopening.
