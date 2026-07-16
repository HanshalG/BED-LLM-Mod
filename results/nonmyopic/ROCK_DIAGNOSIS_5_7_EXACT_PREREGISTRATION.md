# Rock Diagnosis `5-7` Exact Replication Preregistration

Registered: 2026-07-15, before this map/seed's first execution or outcome read.

## Purpose

The confirmed `3-6` LLM result must be replicated on a second independently specified
external environment instance before consolidation. This zero-LLM-call gate verifies
that the depth-over-width mechanism is present on the paper's separate Figure 4 `5-7`
Rock Diagnosis map before a bounded LLM candidate-proposal pilot is allowed.

## Frozen Exact Design

- Source: Araya-Lopez, Buffet, and Thomas (2013), Figure 4 `5-7` map, page 10;
  grid side 7; rock positions `(4,0)`, `(6,2)`, `(2,3)`, `(3,5)`, `(5,5)`; fixed
  left-centre entry `(0,3)`.
- Latent target: the static full five-rock type vector; uniform prior over all 32
  vectors. Motion and sensor likelihood use the same `pomdp_py==1.3.5.1` exact
  RockSample primitives as the `3-6` gate; posterior, EIG, and MAP decode are exact.
- Seed `9173`, 2,000 paired trajectories, horizon 8, K=3, and 10,000 deterministic
  paired percentile bootstrap resamples. The earlier unregistered `5-7` discovery
  used seed 1304 and is not part of this evaluation.
- Arms and CRN: shared-cell d1, exact incremental-EIG d2, and candidate-call-matched
  current-state width. All mechanics, deterministic candidate-cell construction,
  action legality, and outcome keys are identical to the `3-6` exact confirmation.

## Gate

The exact `5-7` gate passes only if d2 has a strictly positive lower paired 95%
bootstrap bound for final-entropy reduction against both shared d1 and call-matched
width, with shared-root, allocation, and legality checks true. A pass permits one
registered <=10-trajectory non-thinking LLM candidate-proposal pilot on this map; a
failure is ledgered as a non-promotion and stops paid `5-7` work.
