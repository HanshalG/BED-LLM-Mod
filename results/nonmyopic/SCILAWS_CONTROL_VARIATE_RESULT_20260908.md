# Analytic variance correction passes one-step fixtures

Previous turn ruled against more tuning of fixed order8 moment weights. This
turn implements the analytic control-variate identity without changing weights,
nodes, likelihoods, latent support or the default uncorrected model.

For each chance node let U be within-family target variance, including observation
noise when requested. Its predictive expectation after a fixed action is analytic.
The corrected continuation is

    Q[V(next)] + E[U(next)] - Q[U(next)].

Only E[U(next)] is exact. Q[V-U] remains a numerical approximation; at deeper
horizons V includes future optimized decisions and their own numerical errors.
There is no assertion of exact multi-step integration or conservative error sign.

## Consistent implementation

ControlVariateMixture is an explicit alternative. Its terminal hook subtracts
the sampled within-family variance and adds the analytic term. Nonterminal and
open-loop chance evaluations use the same signed offset through an optional
shared-planner hook. Every materialized PolicyNode stores quadrature_correction,
and expected_risk equals the weighted child risks PLUS this field. Leaf offsets
are zero. The tree is thus a corrected numerical integral, not an assertion that
uncorrected quadrature masses alone reproduce the displayed value.

Invalid corrections or negative/nonfinite corrected risks fail closed. Existing
models without the hook get offset0. Node charges, time caps and cache limits
remain; caching the four scalar terminal terms is bounded to256 entries. Raw
posterior updates and forecasts do not use this correction.

177 tests pass in17.97s across SciLaws and the ordinary/raw/batched horizon suites.
New tests compare full root values at h1/h2/h3 in both adaptive and open-loop modes
with an independent scalar full-target-moment correction, reconstruct every tree
with its saved offsets, check exact single-family one-step risk and reject NaNs.
Scoped lint passes. This checks implementation consistency, not deep accuracy.

## Public/synthetic one-step audit

Implementation and audit selection pushed at20200f6d before execution. Same eight
task geometries, public prior and artificial action3/value1.7 history, all eight
actions, orders8/16 and old order128 reference as the previous numerical audit.
Same thresholds: maximum absolute value error<=1e-4 and selected-action regret
against that reference<=1e-4. No source observations or new efficacy endpoints.

Artifact SCILAWS_CONTROL_VARIATE_AUDIT_20260908.json SHA256:
6f43d961ba881affc8edcf5e380c72f4fda059a5d0c4aea52b6ecaa3dbb742ca.

| Order | Passing rows | Maximum absolute error | Maximum reference regret |
|---|---:|---:|---:|
| 8 | 16/16 | 7.3706191e-6 | 0 |
| 16 | 16/16 | 7.3640668e-7 | 0 |

Combined fixture audit passes. These are correlated normalized public/synthetic
states, not32 independent source cases. Order128 is a high-order numerical
reference, not a rigorous true integral. The pass does not authorize source or
paid execution, certify root rankings after arbitrary histories, or solve h3 cost.
The earlier optimism and solver-failure records remain banked.

Next: qualify deeper continuation errors and family-bound consistency, then
address search size using error-aware bounds rather than the old optimistic
incumbents. No node/time cap increases, support reduction, source-noise changes
or scientific-threshold relaxation follow from this one-step result. The full
paired policy protocol and source usage review remain incomplete. Useful LLM
proposal/refresh and anticipated-discovery evidence are still required for the
actual project goal; this is only its numerical inference/planning instrument.

No source measurements, LLM calls or paid cost. Processes exited. Authenticated
account and current London ledger unchanged; automation remains paused.
