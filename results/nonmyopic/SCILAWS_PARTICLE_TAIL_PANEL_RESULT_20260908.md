# Explicit-tail panel:32 branches qualify for one-step integration

Frozen implementation/protocol dbc8ce38. Result SHA256
95e53384a587fa90cdbbf776eb22613eae1dd72ec3f615a6839fff201dbce588:
SCILAWS_PARTICLE_TAIL_PANEL_20260908/result.json. All16 shard hashes verified;
process exited normally. All96 candidate plans completed and all independent
references were reused without recomputation.

| Total branches | Cases passing | Worst absolute root error | Worst action regret | Seconds/menu |
| --- | --- | --- | --- | --- |
| 16 | 47/48 | 0.0001142173 | 0 | 0.0907-0.1522 |
| 32 | 48/48 | 0.0000286839 | 0 | 0.1670-0.1977 |

The16-node failure is task2/seed1305/affine. Its threshold miss is preserved;
no rescue from closeness or correct actions. The32-node composite rule passes
the exact frozen full-panel gate. Compared with ordinary32-node correction's
46/48, explicit tail resolution removes the diagnosed failures. It also uses
half the branches of the qualified64-node uncorrected rule.

This is NOT a measured speedup. The uncorrected64-node panel ran0.1582-0.1852s
per menu, while the composite32-node correction takes0.1670-0.1977s. Extreme
quantile inversion adds cost; the reduction is in branching, not current h1
wall time. No performance claim should ignore this. For exhaustive repeats,
32 outcomes imply256 first-level and65536 second-level child beliefs, below
100000 for h2 but not a proof of its5s runtime feasibility. H3 still greatly
exceeds the state limit even though the existing32-node memory admission fits.

## Next Dependency

Keep the32-node rule fixed. Check one-step terminal accuracy at simulated
continuation histories and measure an actual bounded h2 workload before any
deeper experiment. Do not replace an entire h-step Bellman value with this
one-step correction: its exact identity concerns a posterior mean after one
observation. Deeper corrected integration needs its own derivation/reference.
No further initial-history node/boundary tuning is warranted. A runtime failure
would require a genuine search improvement, not relabelled node counts or caps.

This is a software-fixture numerical pass, not source-law calibration or a
non-myopic LLM result. Actual source measurements, deployable horizon opportunity,
useful LLM proposals and paired sealed evaluations remain outstanding.
Thirteen tests pass1.26s; scoped lint passes. Source/model calls0, paid cost0;
authenticated credits245/usage220.376693994/balance24.623306006. London Sept8
ledger spend0. Automation paused; full research goal remains unfinished.
