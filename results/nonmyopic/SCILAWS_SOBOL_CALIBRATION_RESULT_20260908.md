# Sobol posterior calibration passes at the memory-limited count

Frozen6f332162, process exited. Artifact
SCILAWS_SOBOL_CALIBRATION_AUDIT_20260908.json SHA256
82a56106f661fc577020086ca6980446a139e55bf45a41430fa4f08ab5d86fe7.
All144 draws/576 comparisons completed, unchanged observations and gates.

| Particles/family | All-check draw passes | Comparison passes | Current H3 memory fits |
|---|---:|---:|---:|
|32|0/48|90/192|48/48|
|128|42/48|185/192|48/48|
|512|48/48|192/192|0/48|

512-per-family clears the whole declared software calibration screen in both
streams: max standardized mean error.0083313, relative variance error.0294804,
familyTV.00279669, logdensityerror.0272348. IID at the same count passed32/48draws.
128 still fails six draws; its max relative variance error.0812934 exceeds.05.
Do not use185/192 comparisons to rescue that count or select passing cases/seeds.

The full512 calibration pass is meaningful numerical approximation evidence for
the declared initial/one-update fixtures only. It is not source calibration,
full-episode support adequacy,1e-4 planning error, a monotonic-depth result or an
LLM contribution. Current quadratic distance storage plus one depth3 workspace row
still exceeds64MiB, so no planning run is authorized by this result alone.

Next algebraic memory improvement: optional centered-moment terminal risk instead
of a dense particle-by-particle distance matrix. For fixed center c,
Var(T)=E[||T-c||^2]-||E[T-c]||^2, with target weights and the same conditional noise
term. It uses particle-by-target storage rather than particle squared. Independently
test large offsets, concentrated weights, nearzero risks and scalar/batch full roots
before adoption; handle floating-point cancellation conservatively and keep the
existing pairwise default. This changes neither physics nor sample count/gates.
Recompute actual memory preflight after implementation, then separately qualify
unequal-noise integration and full repeated-action planning.

Fourteen sampler/adapter/gate tests passed in.92s; scoped lint passes. IID artifacts
remain untouched. Source/modelcalls/spend0; authenticated balance24.623306006 and
London dailyspend0. No active process; automation paused; full goal unfinished.
