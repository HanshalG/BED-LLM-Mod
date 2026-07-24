# MovieLens Load-Bearing Profile-Dynamics v2 Smoke Result

Date: 2026-07-24

Status: passed. The frozen disjoint formal mechanism gate is authorized.

## Serving Result

Both smoke-only users completed one recorded-rating branch and one exact refresh
replay:

- exactly 10 physical requests;
- zero reasoning tokens, retries, forced exits, parse errors, or runtime failures;
- six valid initial profiles per user;
- six non-copy replacement profiles with nonempty newest-evidence effects per branch;
- eight-profile branch supports after retaining the two best old profiles;
- valid profile-only five-way likelihood matrices;
- total cost `$0.03354118`.

The live endpoint and project ledger agreed on `$21.586319943` remaining after the
smoke.

## Mechanism Audit

The likelihood requests contained profile descriptions and movie metadata but no
observed-history payload. Both users had profile-sensitive initial likelihoods:

| User | Maximum immediate EIG | Initial held-out NLL | Contact-branch NLL | Improvement |
|---:|---:|---:|---:|---:|
| 294 | 0.04098 | 1.41837 | 1.36188 | +0.05649 |
| 327 | 0.02455 | 1.65127 | 1.74888 | -0.09761 |

Every generated description differed exactly from every initial description. Exact
generated/replay overlap was zero, but manual review found stable semantic updates:
user 294 repeatedly shifted toward epic or romantic spectacle over cerebral science
fiction; user 327 repeatedly shifted toward grounded crime and dark humor over
speculative or sentimental films. The diversity is expected from stochastic particle
generation and is not the v1 copy failure.

One branch improved and one worsened. This is not a smoke failure: the frozen formal
gate measures whether the best of four branches improves prediction, whether branch
choice matters, and whether current-support immediate EIG misses the best update.

## Data Handling

Gemma repeated numeric source ratings in one evidence-effect explanation. Raw responses
are therefore retained only in the ignored local `external/` data area. The committed
smoke artifact contains public movie metadata, counts, text hashes, and derived metrics
only. No prompt, response, metric, threshold, or endpoint was changed by this
privacy-only output amendment.

Artifacts:

- `results/nonmyopic/movielens_profile_dynamics_gate_v2/serving_smoke_20260724/SERVING_SMOKE.json`
- `results/nonmyopic/movielens_profile_dynamics_gate_v2/serving_smoke_20260724/run.log`
- private raw responses under ignored `external/private_movielens_profile_dynamics/`
