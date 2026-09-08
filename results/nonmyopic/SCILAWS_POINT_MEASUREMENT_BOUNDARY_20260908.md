# Point measurement boundary

Previous turn: concrete progress via complete public metadata coverage. This
turn implements the evaluator-side accounting and point-query layer needed
before any SciLaws pilot. It does not open the benchmark or authorize inference.

`environments/scilaws/point_measurements.py` accepts exactly request ID, point
coordinates and replicate count. It rejects extra fields (including seed and
where), nonfinite inputs, unknown axes, invalid replicates and out-of-bounds
coordinates before any measurement. Bounds must be verified by the caller;
the public metadata projection alone is not proof of simulator support.

Every accepted request reserves its full row exposure in SQLite before the
first simulator operation. The ledger binds world, paired episode, arm, runtime,
input/target schema, budget and a hash of the private pairing key. Completed
request IDs replay the stored response without another simulator call; they
cannot be reused for different points. An unresolved, crashed or failed attempt
blocks new measurements. Failures keep full reserved exposure, even when only
some replicates executed. No free replacement draw is available.

The evaluator derives separate random seeds from a private key and the paired
world/episode/round/replicate tuple, excluding arm, request ID and query point.
This pairs random streams across policies, not necessarily numerical residuals
at different points with different local noise distributions. Call splitting
is a treatment choice: the eventual protocol must fix the replicate schedule
and measurement budget consistently across arms.

Only matching realized coordinates and finite target observations are returned.
Clipping, missing rows, malformed counters and invalid targets halt the session.
Upstream metadata, seeds, paths and error text are not included in the response.

## Verification

41 combined SciLaws tests pass in2.76s; lint passes. Tests include pre-call
reservation inspection, idempotent restart, changed-binding rejection, budget
exhaustion, paired seed equality, separate replicate/round seeds, reentrant
requests during a pending call, invalid inputs, malformed outputs, and preserved
pending exposure after a simulated process interruption. Integration uses the
pinned TypeI runtime closures on an artificial constant +/-1-noise fixture:
both arms return identical four-observation trajectories, charge exactly four
rows, and replay without another call. No serialized benchmark state is loaded.

## Remaining requirements

This object is not process isolation or a security sandbox. It must remain in
an evaluator process outside the policy filesystem, alongside its key, database
and simulator. A transport broker, process timeout and tested filesystem boundary
are still required for code-enabled policies. The backend call is not itself
time-bounded here; a killed worker leaves its reserved attempt unresolved.
No benchmark loader, task licensing decision or scientific authorization is
provided. The old frozen source-audit result remains unchanged; its source tests
now also exercise this new adapter.

Next scientific dependencies remain the public-prior/noise/target contract,
bounded complete opportunity panel, and fresh proposal/predictive calibration.
Passing these software tests is not evidence of a depth benefit. Account and
London ledger revalidated unchanged at usage220.376693994, balance24.623306006,
zero spend. No model calls, endpoint opening or paid permission; automation
paused and full research goal unfinished.
