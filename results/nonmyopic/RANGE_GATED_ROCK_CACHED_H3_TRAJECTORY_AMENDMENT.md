# Range-Gated Rock Cached-h3 Trajectory Pre-Response Amendment

Status: frozen before any live S0 response or S1 response.

The preregistration assigned fresh S1 seed `24193` and stated that it had not been
used by deterministic dry runs. After freezing the document, a zero-LLM
deterministic reference was accidentally run on seed `24193`. It produced no model
response and caused no code, endpoint, threshold, arm, sample-size, or gate change,
but the seed is no longer pristine under the registered wording.

Seed `24193` is therefore quarantined from the live experiment. The sole amendment
is:

- **live S1 seed: `24195`**

Seed `24195` has not been used by any deterministic or live range-gated trajectory
run. S0 remains seed `24187`. Independent-audit bootstrap seeds remain
`24189`--`24192`. Every other clause of
`RANGE_GATED_ROCK_CACHED_H3_TRAJECTORY_PREREGISTRATION.md` is unchanged.
