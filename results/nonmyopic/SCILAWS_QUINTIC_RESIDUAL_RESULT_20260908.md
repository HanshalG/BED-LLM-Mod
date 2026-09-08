# Quintic residual: three histories pass, initial stress case still capped

Frozen6657e036 before one four-history run.10 focused fit/reconstruction tests
pass in.90s; scoped E4/E7/E9/F lint passes. Internal midpoint target tightens
to1e-5; final2e-5 threshold is unchanged. Polynomial/aliasing tests verify higher
order does not bypass fresh checks or extrapolation restrictions.

Artifact SCILAWS_QUINTIC_RESIDUAL_AUDIT_20260908.json SHA256:
617200658516c57efbaaa31a9362a3b71b42578e97698992ff512b8352d76e34.

| History | Evaluations | Completed roots | Result |
| --- | ---: | ---: | --- |
| Empty | 100001 | 1 | Shared-budget cap |
| Positive | 68243 | 2 | Fresh checks pass |
| Negative | 89221 | 2 | Fresh checks pass |
| Contradictory | 59918 | 2 | Fresh checks pass |

Seven roots pass fresh checks, max normalized error7.49e-6. Empty-history root0
uses60 fitting nodes and has interior estimate .27275526263614397, about-4.73e-8
from the independent single-root switch reference. It still does not yield a
complete initial action ranking. The final-check failure of the cubic candidate
is fixed here, at higher work in completed histories. There is no full-panel
pass, no uniform interpolation certificate and no source/h3 authorization.

## Scope check against the actual experiment

Re-read SCILAWS_REFERENCE_PRIOR_PROTOCOL_20260908.md and reference_prior.initialize.
The planned source policy only starts after shared point-major initial replicates
have set response scaling and conditioned the posterior. Geometry requires14
initial observations for the three-input task,10 for two-input tasks and6 for
one-input tasks. The public zero-label preflight was explicitly runtime-only
and not a posterior-conditioned qualification. These facts predate the present
results and must not be rewritten to rescue an empty-history cap.

The present four-history/two-action synthetic stress test remains incomplete,
and the three conditioned passes cannot be relabeled as the eight-task source
experiment. The candidate also implements h2, not receding h3. Before more
numerical variants, audit the remaining validation layers explicitly: general
stress limits, actual post-initialization mechanics on the frozen full geometry,
independent numerical accuracy, and subsequent source calibration. No hidden
initial labels may be opened merely to bypass the stress failure. Any synthetic
post-initialization check requires its own frozen data-generation rule and all
task coverage; it changes no previous gate or source permission.

Useful LLM proposals, full sequential depth comparisons, matched controls and
anticipated discovery remain unproven. No model/source calls, $0. Account and
Sept8 London ledger unchanged; process exited; automation paused; full goal
unfinished. Do not spend or report efficacy from these numerical diagnostics.
