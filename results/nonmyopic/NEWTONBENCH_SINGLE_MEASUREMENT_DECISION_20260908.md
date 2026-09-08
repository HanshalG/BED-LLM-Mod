# NewtonBench: single-measurement separation diagnostic

Date: 2026-09-08. Source-only diagnostic, zero model calls and $0 new cost.
The all-domain executable was committed/pushed at `09d8bf50` before evaluation.
No designs, priors, noise levels or subsets were searched after the results.

## Question and estimand

At the geometric midpoint of each of the 12 unmodified AutoSciLab wrapper boxes,
how distinguishable are the source's laws from one raw noisy measurement?

Use source law functions directly, their source-defined absolute noise floors,
and relative Gaussian noise level 0.01 (the wrapper default). Compute separately
for the three versions within each of easy/medium/hard, then for all nine pooled.
Each listed source law has equal prior probability. Laws with invalid midpoint
outputs invalidate the corresponding comparison; they are not dropped.

This is an oracle-known finite-source classification diagnostic. It is NOT an
LLM-generated prior, continuous-parameter inference, target prediction risk,
planning-depth test, or a source-level physical validity certificate. Treating
these known source functions as the deployed hypothesis space would remove the
very model-discovery problem we need to study.

For K normal likelihoods with means mu_i and standard deviations sigma_i, the
uniform-prior Bayes classification error obeys:

    P_error <= min(1 - 1/K, sum_{i<j} BC(i,j) / K)

where BC is the Gaussian Bhattacharyya overlap integral. This follows by bounding
the total nonmaximal joint density by pairwise minima, each at most a geometric
mean. A small upper bound certifies easy source-law classification under these
assumptions. A large upper bound does NOT establish difficulty or a planning gap.

## Complete panel

| Domain | Difficulty groups with error upper bound <5%, out of 3 |
|---|---:|
| Gravity | 3 |
| Coulomb force | 1 |
| Magnetic force | 3 |
| Fourier law | 3 |
| Snell law | 0; hard group invalid |
| Radioactive decay | 2 |
| Underdamped harmonic | 3 |
| Malus law | 3 |
| Sound speed | 3 |
| Hooke law | 2 |
| Bose-Einstein distribution | 3 |
| Heat transfer | 0 |

All 12 domains were evaluated, covering 36 difficulty groups. **26 of 36** have
error bounds below 5%; 35 are finite and one is invalid. With all nine laws pooled,
**4 of 12** have bounds below 5%; 11 are finite and one invalid. The pooled low-bound
domains are gravity, magnetic force, Fourier law and sound speed. This 5% tally is
descriptive classification reporting, not the earlier successive-depth gate.

The invalid case is Snell hard v0 at the fixed midpoint. All invalid groups remain
in the report. No alternative angle, narrowed box or finite-only replacement was
tried. The wrapper angle-unit issue remains unchanged.

## Numerical reporting correction

The initial result used floating exponentials, so extremely small normal overlaps
underflowed to displayed zeros. Nondegenerate Gaussian likelihoods do not have
exactly zero overlap. Preserve that original artifact and use the reconciled
report for numerical interpretation.

Reconciliation uses the already saved means/floors only, without any source-law
re-evaluation or new outcomes. Pairwise overlaps are floored upward at 1e-300;
this loosens the error bound instead of making it more favourable. It records
the original artifact's hash and the correction code hash. The counts above are
from the corrected file. These are floating evaluations of an analytic bound,
not interval-arithmetic certificates.

Artifacts:
- `NEWTONBENCH_SINGLE_MEASUREMENT_20260908.json`: original complete source panel.
- `NEWTONBENCH_SINGLE_MEASUREMENT_RECONCILED_20260908.json`: numerical-tail correction.

Both retain raw finite law values, invalid identities, exact source-file hashes,
midpoint inputs and per-group bounds. Ten focused tests passed in 0.38 seconds,
including independent equal-variance normal identities, binary Bayes error
comparison, unit-scale invariance, nonfinite rejection and tail-floor regression.

## Decision

Do not pivot straight into a default-noise, known-source-law NewtonBench depth
sweep. Its finite reference space often saturates after one ordinary observation,
and its raw observation contracts still do not match our chemistry adapter.

This is not proof that predictive-risk gains vanish: rare classification errors
can have large prediction consequences, and continuous parameter uncertainty or
genuinely generated out-of-registry models changes the problem. Nor does it prove
the high-bound domains are good choices; one arbitrary midpoint cannot establish
their optimized one-query or deeper performance. We will not pick those domains
merely because this screening produced higher overlap.

The evidence therefore does not justify buying LLM calls or building a new
NewtonBench adapter yet. The stronger route must justify a large, useful semantic
model space and a nontrivial decision problem together, rather than increase
source noise or hide parameters to manufacture a depth curve. This diagnostic
changes the source-adoption decision, but does not complete the research plan.
Closed endpoints and gates remain unchanged; automation stays paused.
