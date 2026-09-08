# Adaptive fit improves accuracy, shared-budget panel still incomplete

Frozen3859ae35, one unchanged four-history diagnostic. Six fit/gate tests pass
in.88s; scoped E4/E7/E9/F lint passes. Tests explicitly reject oscillation invisible
at fitting/adaptation points but visible on fresh checks. Reused exact values
are cached; final check points cannot overlap adaptation or fitting points.

Artifact SCILAWS_ADAPTIVE_VALUE_AUDIT_20260908.json SHA256:
77f8751a67b9db8cfd931c858735111c7d8d27d0bc2ac52c0a468dc4c0e0bc49.

Five roots finish and all five pass fresh normalized2e-5 checks. Fitting uses
36-64 nodes, versus a global fixed65; the largest case still needs253 total
fit/adaptation/final-check observations. The smaller fits require141-185 points.
Direct adaptive integration of both continuation actions at every new point
remains costly.

-Empty/positive/negative histories hit100001 evaluations after one accepted root.
-Contradictory history completes both roots at96301 evaluations, with interior
  estimates .423677476995815 and .4282209813723023.
-All five fresh check errors lie between1.51e-5 and1.90e-5; normalized reported
  inner errors below8.18e-9. Tail bounds are retained separately.
-Empty-history root0 interior estimate .2727536943454646 differs from the
  independent switch reference by about-1.616e-6. That comparison supports this
  one approximation, not all roots or a uniform-error certificate. Its tail-only
  interval is not expected to include interpolation error and must not be sold
  as a total numerical interval.

No full-panel pass, no h3/source/model authorization. Sample checks remain sample
checks even when every evaluated one passes. Do not rerun only the completed
history or raise the shared cap to turn this into a headline.

## Next bounded diagnostic

Investigate an analytic terminal-risk interval before another fit run. The
known-family revelation risk supplies a lower bound. A feasible best linear
predictor using the next scalar observation supplies an upper bound computed
from current target/observation covariance and observation variance. If that
interval is already narrower than the existing inner error tolerance, its
midpoint can replace an expensive integral with an explicit error bound; wide
intervals still require the existing integration. First derive/test the bounds
independently and inspect their widths on opened diagnostic states. Do not
assume they are tight or deploy a shortcut based on posterior mass alone.

This would preserve the target, full hypothesis space and continuous likelihood,
not lower the interpolation gate. It is only useful if it demonstrably removes
enough work under unchanged accuracy. The source-grounded LLM research stages
remain unfinished. No source/modelcalls, $0; ledger unchanged, process exited,
automation paused, full goal incomplete.
