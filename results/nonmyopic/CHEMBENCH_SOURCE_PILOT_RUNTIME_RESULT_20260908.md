# Source pilot: implemented, blocked by public-prior planning cost

## Frozen Task And Data Boundary

The eight-world protocol is frozen and pushed atca1523d2, before source candidate
responses. SHA2568e1fc9df41d177fa80b2e500c22c663e3a78a980ecedaa355c667116cc2e1d36.
Four mechanisms, four prior parameter particles each, independent hidden-world
seeds, three measurements from four designs, fixed64 prediction targets,
sigma0.15 in scalar log1p-rate space, paired world/round/design noise, six core
arms and separate h1/h2/h3 population-oracle diagnostics. No parameter uncertainty
was reduced to meet numerical limits.

All-pair quadrature failed the first source preflight. Equal-noise upper-envelope
splits fixed the integration-knot count while retaining every particle, passed
the unchanged synthetic qualification, and passed all four source root-rule
checks. These are numerical and prior-only checks, not hidden-world observations.

## Actual Attempts

| Attempt | Terminal phase | Reason | Hidden worlds | Complete worlds |
| --- | --- | --- | --- | --- |
| run-20260908-v1 | public_root_planning | rounded endpoint probability gave infinite quantile | closed | 0 |
| run-20260908-v2 | public_root_planning | global planning resource limit at approximately60s | closed | 0 |

V1 is bound to the original runner at5bf6645d. A regression reproduced its
boundary error, then the nearest-representable-interior correction and exact
pairwise variance identity were tested and qualified before V2. V2 ran from
pushed71337d3f. Both use the same physics, prior, seeds, menu, horizon, branch
count, thresholds and runtime caps. Both failure directories are preserved.

V2 completed h1 and h2 initial plans; h2 required49,408 processed states and
0.744s, versus9.039s in V1. h3 did not return a plan. Total attempt time60.913s.
Only preflight, h1/h2 public root checkpoints and the terminal result exist.
There is no hidden binding or world trajectory. These different-horizon planned
values must not be reported as a matched-budget empirical depth benefit.

## Verification

The full combined numerical/source/runner suite passed95/95 in14.00s after the
precision and terminal-risk corrections. Tests cover eight mocked complete
worlds, all six core and three oracle arms, paired noise indexing, no repeats,
hidden-data exclusion before root qualification, and stopping before the next
world after an incomplete one. Those mocked tests are not source outcomes.

No scientific efficacy gate was evaluated: execution stopped before the endpoint
boundary. This is not an engineering_null or evidence that non-myopia fails on
the source task. It also provides no evidence that the LLM component helps.

## Next Available Work

Profile and accelerate the existing inverse-CDF/Bellman kernels using the public
prior only. Preserve the full16-particle support, three-step contingent search,
64-node rule, complete controls and frozen physical task. Prove equivalence and
rerun numerical qualification before any further implementation-version attempt.
Do not increase the time cap, substitute a greedy tail, shrink the prior, or
launch paid LLM calls. Initial h1/h2 checkpoints are banked evidence; never
overwrite either failed run or present partial policy results as the panel.

Full-plan audit: A has a constructed exact reference and qualified small
continuous numerics; B has its complete runner but zero completed source worlds;
C's new LLM gate and paired pilot have not opened. The research goal remains
active and unfinished. No active process from either source attempt remains.

No paid calls, cluster work, old endpoint reopening, cleanup or automation
reactivation occurred. STATE.md and EXPERIMENTS.md record this as numerical
feasibility progress, not a positive research result.
