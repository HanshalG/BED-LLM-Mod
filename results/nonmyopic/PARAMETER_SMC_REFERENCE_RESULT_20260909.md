# Adaptive SMC improves concentration, but is not a general qualification

Frozen commit 937e539c; all 32 numerical replicates completed once. Independent
1024/2048-point quadrature references agree within 1e-9. No LLM calls or cost.

| Fixture | Particles | Qualified / 8 | Worst log-evidence error | Worst absolute mean error |
|---|---:|---:|---:|---:|
| One parameter | 512 | 6 | .133974 | .001558 |
| One parameter | 2048 | 8 | .082197 | .000737 |
| Four modes | 512 | 2 | .444851 | .384778 |
| Four modes | 2048 | 3 | .229298 | .194125 |

The existing adaptive sampler resolves the severe concentration problem on the
known one-parameter fixture: all 2048-particle replicates satisfy the frozen joint
mean/variance/evidence criteria. Prior importance sampling had qualified 5/8 at
2048 and 0/8 at 32 or 256. This is not an equal-compute comparison: adaptive SMC
uses many more likelihood evaluations.

The four-mode fixture remains unqualified. At 2048 particles, three seeds fail
evidence accuracy and two different seeds fail predictive mean accuracy. All
four quadrants retain particles in every replicate: this is inaccurate mode
weighting, not complete mode loss. Worst predictive mean error is .195 posterior
standard deviations. Variance errors all satisfy 20% at 2048, so checking only
posterior spread would miss the failure. Uniform final weights are not evidence
of independent posterior draws or correct mode probabilities.

Across the four conditions, total evaluated likelihood rows were 77514, 310070,
89687 and 358636 respectively. Total measured sampler time was about .399 seconds.
No cap was hit and no seed was repeated or omitted. The bank records source and
protocol hashes, reference moments, all per-seed diagnostics and mode masses.
The saved artifact does not contain raw particles or the signed evidence estimate;
the regression independently reconstructs moment-based decisions, not raw-particle
replay or a second evidence integration. References remain independently defined.

## Decision

Do not attach this sampler unchanged to arbitrary generated structures or claim
that numerical inference is solved. Do not rerun these seeds with larger caps
and present that as a new scientific gate. The next implementation should make
parameter integration a qualified backend contract: use deterministic converged
integration where low dimensionality makes it practical, and report unresolved
integration uncertainty rather than trusting a single SMC realization elsewhere.
Preserve the existing source bindings and failed scientific cohorts.

This addresses a necessary inference dependency, not the final research result.
After backend qualification, a fresh semantic proposer test must still establish
that new observations produce useful executable structures beyond numerical and
history-blind controls. Only then is joint predictive calibration and ordinary
horizon opportunity worth measuring. Neither the present numerical tests nor
prior reasoning comparisons establish a positive non-myopic LLM result.

Authenticated account usage remains 221.306531939, balance 23.693468061; conservative
London-day spend .88825346 and remaining 4.11174654. Previous turn was a status
restatement (no progress); this turn contributes new numerical evidence. Goal active.
