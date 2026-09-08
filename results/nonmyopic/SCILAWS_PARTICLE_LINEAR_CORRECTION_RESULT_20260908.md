# Particle correction workload: lower-order candidate passes

Frozen implementation/protocol dae7531c. Artifact SHA256
49ccf51cb49f94bd7fd996013519b89859d28cbcfe04618531b99a0bdd0d772a:
SCILAWS_PARTICLE_LINEAR_CORRECTION_AUDIT_20260908.json.

All12 candidates (three fixed first-task histories, four counts) completed.
All independent references were reused from the pinned original workload.

| Branch count | Passing cases | Worst root error | Worst action regret | Seconds per full action menu |
| --- | --- | --- | --- | --- |
| 4 | 3/3 | 0.0000303887 | 0 | 0.0161-0.0184 |
| 8 | 3/3 | 0.0000227974 | 0 | 0.0255-0.0270 |
| 16 | 3/3 | 0.0000158914 | 0 | 0.0445-0.0465 |
| 32 | 3/3 | 0.00000928587 | 0 | 0.0811-0.0844 |

This addresses a specific integration weakness: integrating the full posterior
risk wastes nodes approximating a contribution whose expectation is available
analytically. The linear-predictor correction removes that contribution while
keeping exact particle likelihood updates and conditional target noise.
At4 branches only32 child beliefs are processed for the full eight-action menu.
No particle reduction, endpoint clipping, outcome discretization, action removal,
or resource/gate relaxation was used.

The exact identity does not guarantee lower finite-rule error in general. A
separated-mixture unit fixture has nontrivial residual-tail error even at128
nodes. Independent adaptive integration verifies the identity at1e-8, whereas
finite-rule accuracy is a separate scientific gate. Seven focused tests pass
in0.93s; scoped lint passes.

## Next Required Checks

First extend to all24 public fixtures and both seeds, reusing these three cases,
under the same1e-4 and resource limits. A passing count then needs accuracy at
imagined continuation histories. Do not substitute this one-step evaluator for
all Bellman levels: future optimized value is not a one-step posterior variance.
Any deeper integration correction must preserve that distinction and have an
independent full-root test. Cheaper terminal updates alone do not prove that an
entire depth3 tree fits, nor that deeper receding-horizon decisions improve.

No source outcomes or LLM responses opened. This remains numerical feasibility
work, not evidence of a non-myopic or LLM contribution. Account authenticated at
credits245/usage220.376693994/balance24.623306006; Sept8 spend0. No active process,
no cluster work, automation stays paused, full goal remains unfinished.
