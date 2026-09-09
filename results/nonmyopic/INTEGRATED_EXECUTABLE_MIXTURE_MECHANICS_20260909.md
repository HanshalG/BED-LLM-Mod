# Executable structure-mixture integration bridge

New opt-in IntegratedExecutableBeliefPool reuses the existing canonical law
validation, deduplication and safe numeric interpreter. It does not alter the
existing finite-particle snapshot or frozen scientific routes. moment_snapshot
fits every law from the complete supplied history with the adaptive backend,
then combines equal-prior law evidence, predictive means and within/between-law
variance. It returns conditional mixture moments, not a particle state or a
joint simulator. Observations remain noisy log1p rates.

New hypotheses receive equal prior law mass before full-history likelihood
replay. Renaming/resubmitting a canonical law does not multiply its mass.
Submission order and the inherited particle RNG seed do not affect these fits.
The result remains conditional on the generated pool, not Bayesian correction
for selecting that pool using the same observations.

## Numerical verification

Independent 512-node integration checks the complete two-structure mixture:
constant rate k versus slope rate k*C_A, both k uniform [.5,1.5], observation
sigma .1, forecast at C_A=2. These are opened numerical unit fixtures, not an
LLM experiment or scientific endpoint.

| History inputs | Constant probability | Slope probability | Predictive mean | Latent predictive variance | Conditional log evidence |
|---|---:|---:|---:|---:|---:|
| [1] | .500000 | .500000 | .9022183432 | .0535901851 | .6800006118 |
| [1,2] | .01369556 | .98630444 | 1.1006668742 | .0071655357 | .8889545668 |

Observed values are log1p(input). Mixture moments/evidence match the independent
reference to 1e-7. Scalar-node work is 3790/10024. Refreshing from a fitted
constant-only pool to both structures reproduces a fresh full-history fit exactly;
there is no repeated conditioning or artificial zero start mass for the new law.

Invalid/nonintegrable laws cause the entire mixture to fail; none are silently
filtered or normalized away. Unsupported >2-parameter support fails before any
fit. Tests also exercise total expression-work caps, canonical duplicates and
inherited seed independence. Parameter evaluation caps are shared across laws;
expression/workspace caps include history and target evaluations.

55 tests pass, one skipped (.83s) across the new bridge, adaptive integration,
existing executable pool and structure proposer. The skipped test requires
uninstalled gplearn; no dependency was installed. No API calls or cost.

## Remaining research dependency

Do not run the old 1-8-parameter proposer protocol and silently keep only laws
this bridge supports. It is not yet an unrestricted replacement. A new
prospectively scoped semantic proposal study must justify its parameter domain,
retain all returned hypotheses or fail closed, and compare against numerical,
history-blind and productive symbolic competitors. Its first question is whether
observations induce useful new executable structure, not whether deeper planning
looks better on a hand-constructed pool.

Then require joint predictive calibration, branch-update fidelity and actual
ordinary-horizon opportunity before a depth sweep. Marginal means/variances alone
are insufficient for planning. This bridge addresses inference bookkeeping only;
it does not supply a positive non-myopic LLM result.

Previous turn was progress; this turn implements and verifies the mixture bridge.
Goal active/unachieved. Account unchanged at usage221.306531939/balance23.693468061;
remaining conservative London-day allowance4.11174654. No cluster or automation
changes.
