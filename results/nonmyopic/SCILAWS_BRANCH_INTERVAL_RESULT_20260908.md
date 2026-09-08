# Analytic certification has limited but real coverage

Frozen04cba9ae replay exited successfully. Artifact
SCILAWS_BRANCH_INTERVAL_AUDIT_20260908.json SHA256
087920dd499973128f9a1e7b4c2bb89331904c0ba64cca89b2acd8ac015ed7e0.

All9177 saved numerical references fall inside the analytic interval (allowing
their reported reference error). The candidate-to-both-endpoints test certifies
1786/9177 (19.46%) for each of orders4 and8 at the unchanged1e-4 absolute tolerance.
Widths range1.30574e-6 to.00870358. The two certified counts coincide; neither
candidate is certified on the remaining7391 saved records. This does not imply
those scores are inaccurate: the analytic interval is only insufficiently tight.

This is conditional on the continuous working model and ordinary floating-point
padding, not source calibration or rigorous interval arithmetic. No new numerical
integrals were evaluated. All saved branch identities were verified exactly using
the original eight-thread construction arithmetic; interval algebra used one
thread. Four focused certificate/interval tests passed in1.40s; scoped lint passed.

The19.46% is measured on an ordered, time-censored prefix. Do not extrapolate it to
all branches or claim full-plan feasibility. A next prospective zero-integration
check can examine full-coverage probability-weighted interval widths: an outer root
score combines branch risks with known positive probabilities, so terminal error
contributions add with those weights. This may identify where accurate integration
actually matters and yield a conservative allocation rather than equal effort on
every branch. It must retain all actions/branches and explicitly account for outer
integration error; small weighted terminal uncertainty alone cannot qualify the
continuous h2 objective. No deployment or scientific gate is changed by this replay.

No source measurements/model calls/spend; account usage220.376693994 and balance
24.623306006, London-day spend0. No active process; automation remains paused.
Full LLM-native non-myopic/source/discovery goal remains unfinished.
