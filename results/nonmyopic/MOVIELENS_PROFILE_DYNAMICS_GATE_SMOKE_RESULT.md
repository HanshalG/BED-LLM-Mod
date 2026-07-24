# MovieLens Semantic-Profile Dynamics Serving-Smoke Result

Date: 2026-07-24

Status: serving mechanics passed, but the v1 semantic representation failed manual
pre-formal audit. The 120-request formal gate was canceled before launch.

## Frozen Smoke Result

The first two preregistered users completed one recorded-rating branch and one exact
refresh-prompt replay each:

- exactly 10 physical requests;
- zero reasoning tokens, retries, forced exits, parse errors, or runtime failures;
- six valid initial and refreshed profiles per user;
- valid five-way rating likelihoods whose rows summed to one;
- no source rating, initial history, or held-out item list in the persisted derived
  result;
- total cost `$0.02876341`.

The automated serving-only gates therefore passed exactly as frozen.

## Manual Semantic Audit

Both realized refresh responses copied all six initial profile descriptions exactly.
Exact deduplication consequently left each branch with the same six-profile support it
started with. For user 13 the branch held-out NLL changed from `1.69199` to `1.62257`;
for user 62 it changed from `1.97611` to `2.26365`. Those changes cannot be attributed
to regenerated hypotheses because no hypothesis changed.

The cause is an information bypass in v1: the likelihood prompt receives both the
semantic profiles and the full observed rating history. It can react directly to the
new rating even when the profile generator is a no-op. The smoke therefore does not
establish the load-bearing semantic belief transition required by the project goal.

Initial profile sensitivity was also weak for one of two users. Maximum immediate EIG
was `0.00399` nats for user 13 and `0.07365` nats for user 62, against the frozen formal
sensitivity level of `0.02`.

## Decision

Do not run or reinterpret the v1 formal gate. Close this exact apparatus before formal
endpoint evaluation. A distinct preregistered successor must use a fresh target-blind
user sample, require genuinely revised profile outputs, and hide raw rating history
from the profile-conditioned likelihood prompt so predictive updates can only pass
through semantic hypotheses.

Artifacts:

- `results/nonmyopic/movielens_profile_dynamics_gate/serving_smoke_20260724/SERVING_SMOKE.json`
- `results/nonmyopic/movielens_profile_dynamics_gate/serving_smoke_20260724/RAW_RESPONSES.json`
- `results/nonmyopic/movielens_profile_dynamics_gate/serving_smoke_20260724/run.log`
