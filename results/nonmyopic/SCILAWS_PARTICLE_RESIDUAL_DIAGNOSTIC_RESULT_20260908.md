# Residual-tail diagnosis confirmed on the two failing cases

Frozen code/protocol7b492675. Artifact SHA256
2e16ab1d64231d500c7df5c0c71cf6f7fff6eda9be7601610f7abbedf2aee216:
SCILAWS_PARTICLE_RESIDUAL_DIAGNOSTIC_20260908.json.
All24 action integrals completed,3616 charged evaluations per case including
quantile-boundary evaluation. No previous policy plan or risk reference rerun.

| Case (affine) | Worst saved32-node root error | Residual beyond extreme nodes | Full residual expectation | Maximum independent risk discrepancy |
| --- | --- | --- | --- | --- |
| task0 seed1304 control | 9.28587e-6 | 1.68475e-5 | 4.49038e-5 | 6.11e-16 |
| task2 seed1305 failure | 2.78049e-4 | 2.81393e-4 | 2.89931e-4 | 2.22e-16 |
| task6 seed1304 failure | 2.02076e-4 | 2.06234e-4 | 2.14979e-4 | 2.22e-16 |

Residual columns refer to the worst-error action in each case:7,6,5 respectively
(zero-based). Beyond-node fractions in the two failures are about97.05% and95.93%.
The independently integrated advantage deficit equals the saved root error to
roundoff. This verifies the correction identity on these actual failing mixtures,
and attributes their error to residual quadrature, dominated by outer outcomes.
It is not a posterior-update bug or a risk-reference discrepancy in these cases.
Probability was not literally dropped: quadrature nodes represent intervals,
but the outer intervals were represented inadequately for this integrand.

The two failures have far more of their nonlinear correction concentrated in
outer outcomes than the passing control. Thus doubling ordinary quantile counts
barely affected the relevant contribution. Full-density residual integration,
with explicit splits and analytic omitted-tail bounds, resolves it well within
the diagnostic resource caps. This does not make the full adaptive integral a
qualified cheap terminal evaluator inside a large tree.

## Next Candidate

Use explicitly separated central and tail probability intervals for the residual,
with full posterior likelihoods and accounted probability weights. Freeze a
small composite rule and total branch budget before its results; verify it
against independent integration on adversarial separated/unequal-noise mixtures
and the full48-case public panel. Do not suppress the tails, clip the slope,
alter the particle posterior, or count only central nodes. Simulated-history
and full-Bellman accuracy remain necessary even after any new one-step pass.
The current failed correction panel stays failed. This diagnostic authorizes
no source/LLM/deep policy run and changes no scientific threshold.

Two new tests passed in0.79s, scoped lint passed. Selection was retrospective
and failure-informed; the table is a mechanism diagnostic, not a population
estimate. Process exited normally, source/model calls0, paid cost0, account and
London ledger unchanged. Automation paused; full research goal unfinished.
