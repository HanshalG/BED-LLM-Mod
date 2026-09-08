# Analytic branch-score certification feasibility

Before replay: use exactly the9177 saved completed records bound by1e5cb5d5.
Reconstruct and exactly verify branch identity. Compute the existing known-family
lower and best-linear-prediction upper bounds; compare both saved candidate scores
to both endpoints. A score is certified to the existing1e-4 absolute tolerance only
if its maximum distance to either bound is<=1e-4. Do not substitute closeness to the
saved numerical reference. Record all intervals and successes/failures.

This uses no new integration or source outcomes. Fixed initial tables and geometry
are unchanged. Saved reference containment is a regression check using its reported
error estimate, not proof of bound validity. Existing bound mathematics and tests
provide that working-model justification (with ordinary floating-point padding,
not interval-arithmetic certification). Prior9177-record ordered-prefix limitations
remain: no full case, source calibration, h2/h3 or deployment claim is authorized.

The purpose is to establish whether a conservative analytic shortcut has enough
coverage to merit a complete prospective implementation, not to relax any gate.
