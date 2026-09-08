# IID joint-particle calibration: no feasible qualified count

Frozen8f982ead, process exited. Artifact
SCILAWS_PARTICLE_CALIBRATION_AUDIT_20260908.json SHA256
55cd490581ec95fba1c538f6ec0425fa31a705c6d623211d29c93cd884b2dff5.
All144 draws and576 initial/update comparisons completed; no source labels.

| Particles/family | Draws passing all4 checks | Individual comparisons passing | H3 workspace fits |
|---|---:|---:|---:|
|32|0/48|8/192|48/48|
|128|4/48|79/192|48/48|
|512|32/48|170/192|0/48|

No count passes the complete frozen screen. Maximum standardized target-mean error
is.21695/.10474/.05469; relative target-variance error.23352/.14181/.07830;
familyTV.13980/.06321/.04006; log-density error.68042/.22658/.15362 at32/128/512.
Minimum ESS fractions.30485/.32806/.37000 pass the.1 floor. These data do not show
particle collapse under these updates; ordinary finite-sample approximation remains
too inaccurate. The screen does not establish behavior over a full real episode.

512-per-family fits adapter construction but fails the separate64MiB depth3 batch
workspace estimate at16 branches. Increasing IID samples therefore cannot simply
be adopted as a fix. Do not drop failed seeds/cases, loosen gates or count successful
individual comparisons as successful draws. No depth, source or LLM run authorized.

Next prospective route: variance-reduced joint posterior sampling (scrambled Sobol
normal/inverse-gamma transforms with explicit family weights), retaining the same
posterior and low-count memory limits. First validate the joint transformation,
reproducibility, parameter/noise dependence and finite-tail handling. Then freeze
the same full count/seed/observation/gate panel before its new results. IID failures
remain banked; QMC is not presumed to pass or to provide exact Bayesian inference.
Planner risk/memory optimization is separate and must not change the target loss.

Ten focused adapter/gate tests passed in.83s; scoped lint passed. Account usage
220.376693994,balance24.623306006,London dailyspend0. No active process, automation
paused. Useful LLM proposals, validated non-myopic planning and source efficacy
remain unproven; the full goal is unfinished.
