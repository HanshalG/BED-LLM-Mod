# Zendo Filtered-Particle Confirmation Result

Date: 2026-07-25

Status: **failed closed before branches, scorers, or endpoints**.

Run: `zendo-filtered-particle-confirmation-20260725T075054Z`

All seven initial responses were served and checkpointed. Five had strict JSON
envelopes. `upsilon` and `xi` each contained one complete JSON object followed by
one extra `}` character, so `json.loads` rejected them as extra data.

Among the five strict envelopes:

- `iota`, `omega`, `nu`, and `psi` had 12 valid AST particles;
- `kappa` had 11 valid AST particles and one invalid cross-field sample, which
  the prospectively frozen filter handled as intended.

Particle filtering cannot apply to a malformed top-level envelope. The run
therefore stopped after seven requests, before any branch refresh, scorer, hidden
predicate, root endpoint, or policy comparison.

There was no first-object extraction, character deletion, response replacement,
reissue, or partial continuation. Per the preregistration, no further Zendo
parser or serving amendment will be made on these tasks.

## Usage

- Requests / HTTP attempts: `7 / 7`
- Prompt / completion / reasoning tokens: `3,140 / 4,495 / 0`
- Retries / forced exits: `0 / 0`
- Cost: `$0.075275`
- Branch / scorer / endpoint calls: `0 / 0 / 0`

The multi-rule efficacy hypothesis remains unmeasured. The `mu` one-task positive
is retained only as exploratory mechanism evidence, not a replicated claim.

## Reproducibility

- Public failure:
  `results/nonmyopic/zendo_filtered_particle_confirmation/RESULT.json`
- Raw checkpoint SHA-256:
  `70b2c0a7530c3ec2ad4c38708e99d8ed9c60ecab32602c96a1d228bde6948f61`
- Failure artifact SHA-256:
  `5282867115396aa437cdfc58b71236b86193ff8be86d78a74cda630e6e42af59`

Raw responses remain untracked.
