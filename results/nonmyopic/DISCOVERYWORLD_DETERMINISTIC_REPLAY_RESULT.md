# DiscoveryWorld Deterministic Replay Result

## Decision

The preregistered deterministic-wrapper gate **passes exactly**. Two fresh
replays agree on every frozen field for all eight DiscoveryWorld scientific
themes.

This removes the simulator replay blocker for paired counterfactual evaluation.
It does not establish a BED prior, a non-myopic opportunity, an LLM policy
advantage, or scientific-task efficacy.

## Frozen Protocol

- Official source commit:
  `fd591323920be0d3786ef350955de1945aa571e5`
- Difficulty: `Challenge`
- Scenario seed: `0`
- Themes: all eight official scientific themes
- Trace: initial observation plus five deterministic actions
- Wrapper:
  - reset Python and NumPy process RNGs from the frozen trace seed;
  - seed each object's private RNG from the interface version, world seed, and
    object UUID; and
  - instantiate a fresh official `DiscoveryWorldAPI` for each replay.

The gate required exact equality of process seed, actions, action results, all
six UI hashes, all six scorecard hashes, normalized terminal score, completion,
and successful completion for all eight themes. No subset or relaxed field was
allowed.

## Result

| Theme | Exact | Observations | Actions |
| --- | ---: | ---: | ---: |
| Combinatorial Chemistry | yes | 6 | 5 |
| Archaeology Dating | yes | 6 | 5 |
| Plant Nutrients | yes | 6 | 5 |
| Reactor Lab | yes | 6 | 5 |
| Lost in Translation | yes | 6 | 5 |
| Space Sick | yes | 6 | 5 |
| Proteomics | yes | 6 | 5 |
| Rocket Science | yes | 6 | 5 |

Summary: `8 / 8` exact themes.

Audit artifact:
`results/nonmyopic/discoveryworld_deterministic_replay_audit.json`

SHA-256:
`8ff0e6d9a0fa7607370577b10292514bc3461461159c1c3c8140287e87ca72a6`

## Interpretation

The source's unseeded per-object RNGs were sufficient to explain the earlier
same-seed divergence. Deterministically deriving those private seeds makes the
tested traces replay exactly without changing scientific mechanics, action
semantics, observations, or scoring.

This pass authorizes only a separately frozen, zero-model-call audit for a
native lower-immediate/higher-terminal scientific opportunity. A paid policy
run remains blocked until that audit also defines a non-seed latent-law prior
and keeps semantic hypothesis or experiment proposal load-bearing for the LLM.

OpenRouter calls/cost: `0 / $0`. OatML jobs: `0`.
