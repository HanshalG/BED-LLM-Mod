# Linear-baseline residual: partial improvement, not panel qualification

Frozen3b6d7321, one four-history run.11 focused fit/reconstruction/interval tests
pass in1.40s; scoped E4/E7/E9/F lint passes. Reconstructed normalized score error
equals minus residual error by algebra and test. Signed residual estimates are
not clipped; reconstructed action risks must remain nonnegative and finite.

Artifact SCILAWS_LINEAR_RESIDUAL_AUDIT_20260908.json SHA256:
fa4d732e9504d1c0f58b3b8b28e0213a6713ff4a6c1d255594aa320efecf2544.

| History | Evaluations | Result |
| --- | ---: | --- |
| Empty | 100001 | First root passes, second budget cap |
| Positive | 53304 | Both roots pass fresh checks |
| Negative | 65297 | Root0 fresh check fails; root1 passes |
| Contradictory | 50911 | Both roots pass fresh checks |

Negative-history root0 max normalized error is2.2621870874545234e-5 versus the
unchanged2e-5 threshold. It emits no score. This near miss remains a failure;
do not loosen the gate or use the fresh points to refit that banked run.
Six completed roots pass, one completed fit fails, and one is incomplete.

The positive/contradictory histories need19-22 fitting nodes per action compared
with36-40 for preceding accepted full-value fits. Empty-history root0 still
needs50 nodes. Its interior risk .272755649888538 differs from the independent
switch reference by3.40e-7; no complete empty-history action ranking follows.
This supports a useful representation change, not uniform accuracy or a full
synthetic reference. Tail-only intervals still exclude interpolation uncertainty.

## Remaining decision

Retain the analytic baseline in further numerical investigation, but keep this
specific cubic residual candidate partial/failed. Two distinct issues remain:
the initial mixed belief's approximation cost and a final-check miss after
midpoint-based adaptation. Any descendant must prospectively address both under
the same final threshold and total budget, with fresh validation untouched by
refitting. Smooth individual action residuals permit investigating higher-order
local approximation; internal refinement should have an error margin below the
final acceptance threshold, not use the entire allowance before fresh checks.
This is a design direction, not authorization to replay or rescue the failed run.
Require adversarial approximation tests before another diagnostic; do not assume
more polynomial order or a smaller internal tolerance will suffice.

The numerical pilot is still not complete, and source usage review, real source
measurements, useful LLM-generated models, paired depth/compute controls and
anticipated discovery remain unresolved. No LLM/source calls, $0; ledger/account
unchanged, process exited, automation paused, full goal unfinished.
