# Predictive coordinates do not resolve nested cost

Code pushed b702f18e before a single unchanged four-history reference panel.
Optional predictive_coordinates defaults false. For each integral use the
predictive mixture mean and standard deviation as center/scale, integrate over
the full real standardized line, and multiply by the exact scale Jacobian.
No observation distribution, tolerance, action, outcome or resource limit changes.

15 focused tests pass in1.13s, scoped E4/E7/E9/F lint passes. Tests cover all
four terminal histories against raw coordinates, the integral of a narrow normal
density centered far from zero (Jacobian check), density equality and invalid
coordinates. Maximum saved h1 difference from the raw-coordinate result is
1.82e-13. Coordinate conditioning is numerically consistent here, not a new
scientific result or proof of nested convergence.

Artifact SCILAWS_ADAPTIVE_COORDINATE_AUDIT_20260908.json SHA256:
76f9613d14b7d4f1694c3f2cd4dcdae697072d3425dae2fa703d71939cbce006.

H1 all4 complete with480/300/360/300 evaluations, compared with300/240/420/300
in the preceding raw-coordinate run. H2 all4 still terminate at100001 evaluations.
Thus the change does not generally reduce evaluations and does not qualify h2.
Do not describe it as a successful depth-two acceleration or deploy it by default.

## Stop and diagnose work allocation

Several exact algebraic changes have now preserved h1 while failing to complete
the nested reference. Stop full-panel retries based on conjectured speedups.
Next inspect allocation within one already-opened synthetic case: outer callback
count, terminal integrations per outer state, their evaluation/error distribution,
and changes in the minimizing continuation action. This should distinguish broad
inner integration cost from outer refinement around policy-switch kinks or tails.
Use bounded diagnostic instrumentation and retain all limits; do not claim a
cause before measuring it. Only a materially supported new integration strategy
justifies another full panel. Numerical reference qualification remains separate
from source opportunity, LLM proposal quality and paired horizon efficacy.

No source observations/modelcalls, $0. Account/Sept8 London ledger unchanged;
audit process exited, automation paused, full research goal unfinished.
