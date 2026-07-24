# MovieLens Load-Bearing Profile-Dynamics v2 Formal Result

Date: 2026-07-24

Status: failed the frozen conjunction. The exact v2 apparatus is closed before policy
or depth evaluation.

## Execution

The disjoint 12-user formal gate completed:

- 12 initial profile generations;
- 12 profile-only initial likelihood requests;
- 48 recorded-rating profile refreshes;
- 48 profile-only branch likelihood requests;
- exactly 120 physical requests;
- zero reasoning tokens, retries, forced exits, parse errors, or runtime failures;
- six non-copy generated profiles and eight total branch-support profiles for every
  branch;
- total cost `$0.44975180`.

The live endpoint and project ledger agreed on `$21.136568143` remaining after the
run. Raw responses remain in the ignored local data area. Their SHA-256 is
`386124b2d66f91456013c36bd07374e8c8b523c27ddf7ab34ea1f18b22186e22`.

## Frozen Gates

| Condition | Required | Observed | Pass |
|---|---:|---:|:---:|
| Complete requests, users, branches, zero reasoning | exact | exact | yes |
| Mean oracle held-out NLL improvement | >= 0.05 | 0.07232 | yes |
| Users improving by at least 0.05 | >= 6/12 | 5/12 | **no** |
| Users with branch NLL spread at least 0.10 | >= 6/12 | 6/12 | yes |
| Mean immediate-EIG held-out NLL regret | >= 0.03 | 0.04779 | yes |
| Users with immediate-EIG regret at least 0.05 | >= 4/12 | 4/12 | yes |
| Mean maximum immediate EIG | >= 0.02 | 0.02163 | yes |
| Users with maximum immediate EIG at least 0.02 | >= 8/12 | 5/12 | **no** |

The global Spearman correlation between immediate EIG and negative realized branch
NLL was `0.24153`.

## Per-User Endpoint

Positive improvement means the oracle best regenerated-profile branch beat the initial
profile support.

| User | Max EIG | Oracle improvement | EIG regret | Branch spread |
|---:|---:|---:|---:|---:|
| 378 | 0.02222 | +0.39634 | 0.00000 | 0.10193 |
| 387 | 0.01776 | +0.20871 | 0.07945 | 0.33622 |
| 416 | 0.01841 | -0.04905 | 0.00000 | 0.15992 |
| 450 | 0.06851 | -0.11287 | 0.03039 | 0.07109 |
| 470 | 0.03576 | -0.04986 | 0.09479 | 0.09479 |
| 488 | 0.00512 | +0.13468 | 0.04920 | 0.04920 |
| 533 | 0.02922 | +0.01167 | 0.04567 | 0.07405 |
| 537 | 0.00353 | +0.15359 | 0.00000 | 0.09172 |
| 580 | 0.01185 | -0.00365 | 0.18027 | 0.18027 |
| 650 | 0.02006 | +0.12995 | 0.00772 | 0.03876 |
| 676 | 0.01635 | +0.04479 | 0.00000 | 0.21639 |
| 699 | 0.01072 | +0.00351 | 0.08600 | 0.35763 |

## Interpretation And Decision

V2 establishes that semantic profile regeneration can be load-bearing: likelihood
prompts had no history bypass, branch choice changed held-out prediction, mean oracle
improvement passed, and current-support EIG incurred positive realized regret. The
effect is not robust enough for the frozen claim, however. Large gains are concentrated
in a few users, and only 5/12 initial supports make the four candidate ratings
sufficiently distinguishable.

Do not tune thresholds, select the five responsive users, or run a v2 policy. A
scientifically distinct successor would need to improve the belief representation
before seeing outcomes, for example by generating candidate-contrastive semantic
profiles rather than broad taste summaries. It must use the remaining fresh eligible
users and pass a new preregistered mechanism gate before any non-myopic planner.

Artifacts:

- `results/nonmyopic/movielens_profile_dynamics_gate_v2/formal_seed24303_20260724/GATE.json`
- `results/nonmyopic/movielens_profile_dynamics_gate_v2/formal_seed24303_20260724/run.log`
