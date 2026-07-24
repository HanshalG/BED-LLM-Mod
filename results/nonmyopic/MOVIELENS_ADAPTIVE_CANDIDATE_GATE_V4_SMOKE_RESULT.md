# MovieLens Adaptive-Candidate v4 Smoke Result

Date: 2026-07-24

Status: serving mechanics passed. A frozen formal sensitivity futility screen is
authorized before any branch outcome is read.

The two smoke users completed exactly 10 requests with zero reasoning tokens, retries,
forced exits, parse errors, or runtime failures. Both 24-movie likelihood responses,
top-EIG branches, non-copy profile refreshes, eight-profile supports, and exact refresh
replays completed. Cost was `$0.04622408`.

Adaptive selection did not remove the sensitivity warning:

| User | Best-of-16 EIG | Selected movie | Held-out NLL improvement |
|---:|---:|---|---:|
| 113 | 0.01454 | Scream (1996) | +0.05535 |
| 130 | 0.00553 | The Empire Strikes Back (1980) | -0.21335 |

Both maxima are below the frozen formal `0.02` threshold. Before any formal-user
response, the formal process was amended to stop after its 24 required initial calls
unless the unchanged mean and 8/12 sensitivity conditions pass. No candidate or
held-out rating is read at that stopping point. If sensitivity passes, the same process
continues unchanged to the original 120 requests.

Raw responses remain ignored/private. The committed serving artifact contains only
hashes, counts, public movie metadata, and derived metrics.

Artifacts:

- `results/nonmyopic/movielens_adaptive_candidate_gate_v4/serving_smoke_20260724/SERVING_SMOKE.json`
- `results/nonmyopic/movielens_adaptive_candidate_gate_v4/serving_smoke_20260724/run.log`
