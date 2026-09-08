# Terminal interval is valid but too wide to skip integration

Frozen9f7f2142 replay, no new integrals. Three focused tests pass in.92s;
scoped E4/E7/E9/F lint passes. Tests independently integrate the loss of the
linear predictor and verify single-family coincidence of upper/lower bounds.

Artifact SCILAWS_LINEAR_INTERVAL_AUDIT_20260908.json SHA256:
912ba6593403539dea0e8bbc0209a06891a7fef2341264b1bd2bfb880009489a.

All560 completed saved inner estimates lie in their analytic intervals allowing
their reported numerical integration errors. Zero intervals have half-width
<=1e-8; minimum full width is6.63084e-5. Potential saved evaluations: zero.
Therefore do not implement the proposed midpoint shortcut or loosen its tolerance.
The lower bound is too far from the true risk to certify that shortcut here.

## Measured alternative use

Offline matched comparison with the saved trace shows that the upper-bound gap
(linear predictor risk minus adaptive risk) is smaller than the lower-bound gap
(adaptive risk minus family-revelation risk) in all560 rows. With the diagnostic
normalization1+y^2 using raw first observation y, upper-gap median/max are
.00167957/.01578366 versus lower-gap .01377524/.13190889. This is not the
standardized-z interpolation normalization, and smaller magnitude alone does not
prove smoother interpolation or uniform approximation error.

It does motivate testing the analytically computable linear-predictor risk as
a value-function baseline: interpolate only the remaining nonlinear prediction
advantage, then subtract it from the exact linear risk before minimizing over
actions. The conditional linear baseline can be evaluated cheaply at any outer
observation. Keep the previous full-value fit banked; the residual candidate must
preserve the shared cap, fresh-check rule, error threshold and tail accounting.
Test actual reconstructed action-value errors, not just apparent residual size.
No deployment follows from these descriptive magnitudes.

This is a changed approximation representation, not a claim the bound interval
can skip integration. It still needs new training/check integrations and could
fail. No source/LLM endpoint is authorized. Account/Sept8 ledger unchanged,
$0, no calls/new integrals in this replay, no process running, automation paused,
and the full research goal remains unfinished.
