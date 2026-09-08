# Full particle integration panel: one-step pass, deep feasibility unresolved

Frozen code/protocol: d38774d8. Result SHA256:
06f7f6c21ceda8a34ee5cc96f7199d3da677179db241d9c6b1f20e5288e0353d.
Artifact: SCILAWS_PARTICLE_INTEGRATION_PANEL_20260908/result.json.
All 15 newly written task/seed shard hashes verified. The three pinned workload
cases were reused exactly; 45 cases were newly evaluated. Process exited normally.

| Quantile branches | Cases passing | Maximum root error | Maximum action regret | Plan seconds |
| --- | --- | --- | --- | --- |
| 32 | 30/48 | 0.000498621 | 0.0000415383 | 0.0836-0.1020 |
| 64 | 48/48 | 0.0000910986 | 0 | 0.1582-0.1852 |

All 384 root references completed, with 3024 integrand evaluations per case;
maximum numerical error estimate plus analytic tail bound was 5.752e-11.
All 96 candidate plans completed. Count64 passes the frozen one-step gate on
every public fixture and both seeds. Count32 fails; choosing mostly correct
actions does not rescue its root-value error. These comparisons are against
the same finite particle posterior, not the underlying continuous posterior or
source response law. The separate Sobol calibration is a finite diagnostic,
not a uniform approximation guarantee over future histories.

## Why This Does Not Yet Enable Depth

The actual planner memory-admission probe passes at32 branches but fails at64
for depth3 under64MiB. The probe stops at the batched branch boundary; it does
not execute a depth experiment. An initial unit-test probe intercepted the
scalar rather than batched interface and hit the5s test cap; this was corrected
before freezing and running the panel. Final focused tests22/22 in1.25s; lint
passed.

There is a second, independent barrier: with8 actions and64 outcomes, exhaustive
depth2 with repeats has512 first-level child beliefs and262144 second-level
child beliefs, already above100000 states. Fixing memory alone cannot qualify
this planner. Horizon-one accuracy also does not establish tail-node or
multistep decision accuracy. No caps, thresholds, support, or action menu changed.

## Next Decision

Do not run an unchanged exhaustive depth sweep or spend on an LLM yet. The next
candidate must address *both* search cost and accumulated integration error:
an explicitly bounded/adaptive search or a separately validated multilevel
integration scheme, with independent full-root accuracy checks and charged
work. Test feasibility on the opened synthetic fixtures before another full
panel. Reusing unqualified low-order tail nodes, hiding vectorized state counts,
or merely enlarging memory would not solve the scientific dependency.

This is useful numerical infrastructure, not a publishable non-myopic LLM
result. Source-law calibration, a real horizon opportunity, useful LLM model
proposals, paired controls and sealed evaluation all remain outstanding.
Zero source measurements, inference calls and paid cost. Authenticated account
245 credits /220.376693994 usage /24.623306006 balance; London Sept8 ledger
spend0, remaining5. Automation stays paused; the full goal remains unfinished.
