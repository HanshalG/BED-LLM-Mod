# Prior-only particles do not reliably resolve a known posterior

Frozen9d8238e3/protocol in PARAMETER_INTEGRATION_REFERENCE_PROTOCOL_20260909.md.
Numerical fixture only: the correct one-parameter law was supplied, with no LLM
or structure ambiguity. Independent log-parameter Gauss-Legendre1024/2048 agreed
within1e-9 on means, variances and log evidence. No scientific endpoint opened.

| Prior draws/law | Qualified seeds | ESS range | Worst mean error | Worst absolute log-evidence error |
|---|---:|---:|---:|---:|
| 32 (default) | 0/8 | 1.00-1.09 | .249294 | 37.179083 |
| 256 | 0/8 | 2.14-6.93 | .012427 | 1.487509 |
| 2048 | 5/8 | 21.88-31.85 | .004469 | .252269 |

Qualification required maximum mean error<=.01, log-evidence error<=.1, ESS>=10
for each predeclared seed0..7. All seeds and counts retained, no extension.
Reference log evidence6.5123736055; target latent log1p-rate predictive means
.231838/.714652/1.283254/1.827185 with nonzero posterior variances.
Full per-seed results are in PARAMETER_INTEGRATION_REFERENCE_AUDIT_20260909.json.

## Diagnosis

ExecutableBeliefPool correctly weights its sampled particles, but its fixed
prior sample can miss a narrow posterior. Default32 draws over the broad log prior
collapse almost entirely onto one draw even for a single unknown coefficient.
The large evidence error matters separately from mean prediction: relative model
weights could be badly distorted even when one mean happens to look reasonable.
No individual model-selection error was measured by this one-law fixture.

This distinguishes arithmetic correctness from approximation adequacy. The
previous unit tests verified the former against the same finite draws. The new
independent integral tests the latter. ESS alone is not proof of adequate coverage,
and increasing particle count alone is not established as a sufficient fix.

The log-uniform[.01,100] prior and six observations are a prespecified stress
fixture, not a claim that every real history has these errors. A one-dimensional
pass would not certify eight-parameter structures. This does not explain the
constant-only Luna outputs: that earlier correction interface used a different
analytical global-scale fitter. It reveals a separate obstacle to the proposed
reuse of parameterized structure inference.

## Engineering and research decision

Do not place a new expensive proposer/planner behind the unqualified default or
silently change old runs to use more draws. Keep the old fitter and results intact.
The next implementation should qualify posterior-adapted integration (tempering
and rejuvenation or importance correction with explicit proposal density), with
independent references and multiple modes before use. Evidence normalization must
remain valid; fitting a MAP and treating equally weighted local samples as prior
particles would bias structure selection. Fresh and old structures must both replay
the same public history from their declared priors, without double counting.

Reuse existing parameterized syntax and likelihood evaluation, but separate the
integration backend so old response-bank replay bindings remain intact. New code
should report ESS, evidence stability and predictive convergence, retain all mode
mass it represents, and stop before policy deployment if numerical checks fail.
Benchmark posterior adaptation first on low-dimensional independent integrals,
then on multidimensional identifiable and multimodal fixtures. Those are mechanics
qualifications, not substitutes for the LLM-native BED result.

Only a qualified fitter can support the next new-source semantic proposal gate:
same-history numerical fitting for real, blind/shuffled and symbolic proposals;
then independent future-response/updater fidelity and ordinary-horizon controls.
No closed source cohort, paid route, or old failed gate is reopened here.

## Verification and accounting

The first pytest attempt failed because repository conftest installs a lightweight
torch stub without Tensor, which SciPy's array helper inspected. The standalone
audit was then run and completed before that test-harness conflict was corrected;
this ordering is recorded rather than claimed to have been a clean preflight.
Test-only isolation now temporarily removes/restores the stub. No numerical
implementation, fixture, output, or threshold changed, and the audit was not rerun.
The reference test plus23existing fitter tests now pass:24tests in .46s.

Previous/current turns progress. APIcalls0/cost0, account usage221.306531939,
balance23.693468061, London-day conservative remaining4.11174654 including
old .04 uncertainty. Full scientific goal active/unachieved.
