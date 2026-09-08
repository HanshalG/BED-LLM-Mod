# Whole-mixture one-step reference workload

Frozen78925ca2, process exited. Artifact
SCILAWS_PARTICLE_INTEGRATION_AUDIT_20260908.json SHA256
4597f7e59916cd633fcd0bddb0c7eac96d1b9239e1c6c55aa2b8dc6722ba1b57.
All3 first-task references complete all8actions with3024 integrand calls each.
All15 candidate plans complete. Reference mass/error/tail gates pass.

| Branches | Zero max root error | Affine max root error | Quadratic max root error | Cases passing |
|---|---:|---:|---:|---:|
|4|2.28814e-4|3.26388e-3|1.42130e-3|0/3|
|8|7.26972e-5|1.04553e-3|4.39197e-4|1/3|
|16|2.12059e-5|2.93636e-4|1.31226e-4|1/3|
|32|5.38127e-6|7.97552e-5|3.92754e-5|3/3|
|64|1.18571e-6|2.20902e-5|1.06967e-5|3/3|

32-branch candidate times.089-.091seconds,64-branch.164-.175seconds on this workload.
These are actual one-step planning calls on synthetic posterior particles, not
source experiments or h3 timings. Adaptive reference error estimates and probability
mass checks are not rigorous protection against every unresolved narrow feature.
Three focused reference tests pass in.69s, scoped lint passes; small unequal-noise
agreement is checked against the independent componentwise adaptive reference.

Next freeze the full24fixture/twoseed integration panel at the calibrated512/family,
preserving every action and reporting32/64 separately. Retain the cheaper-setting
failures; do not deploy16branches from the zero-case pass. Check memory at the actual
candidate branch count, not the earlier16-branch estimate, before deeper runs.
One-step integration accuracy is separate from particle approximation and deeper
decision error. Source likelihood calibration and useful LLM-generated models still
remain untested; no scientific efficacy or source/LLM authorization follows.

Account usage220.376693994,balance24.623306006,London dailyspend0. No source/modelcalls,
no active process; automation paused and full goal unfinished.
