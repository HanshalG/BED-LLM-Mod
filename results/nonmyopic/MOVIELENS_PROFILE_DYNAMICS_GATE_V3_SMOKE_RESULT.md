# MovieLens Candidate-Contrastive v3 Smoke Result

Date: 2026-07-24

Status: failed closed. No v3 formal, scorer, policy, or depth run is authorized.

## Execution

The smoke made exactly 10 physical requests with zero reasoning tokens, retries,
forced exits, or runtime failures. Both initial supports, both original refreshes, and
one of two exact refresh replays satisfied the candidate-contrastive schema. Total
cost was `$0.03587022`.

The stricter project ledger left `$21.100697923`; the provider endpoint lagged at
`$21.117164693`.

## Frozen Failure

User 26's exact-prompt refresh replay collapsed all six Toy Story reactions to
`avoid`. Its per-movie labels were:

- Contact: `uncertain, avoid, avoid, uncertain, avoid, appeal`;
- Return of the Jedi: `appeal, appeal, uncertain, appeal, uncertain, appeal`;
- Scream: `appeal, uncertain, appeal, appeal, appeal, appeal`;
- Toy Story: `avoid, avoid, avoid, avoid, avoid, avoid`.

The frozen parser required at least two reaction labels for every design movie across
the six profiles. User 63's replay and both original refresh responses passed, so this
is stochastic constraint instability under an exact repeated prompt.

## Decision

Close the exact free-form candidate-contrastive apparatus. Do not relax the diversity
rule, retry the failed replay, or launch the formal gate. A distinct successor would
need to control contrast allocation structurally before generation rather than ask one
free-form response to discover and obey a global support-level diversity constraint.

Artifacts:

- `results/nonmyopic/movielens_profile_dynamics_gate_v3/serving_smoke_20260724/SERVING_SMOKE_FAILURE.json`
- `results/nonmyopic/movielens_profile_dynamics_gate_v3/serving_smoke_20260724/run.log`
- private raw responses under ignored `external/private_movielens_profile_dynamics/`
