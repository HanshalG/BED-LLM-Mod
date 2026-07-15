# Rock Diagnosis Discovery Screen

## Status

**Exploratory discovery only, not preregistered and not confirmation evidence.**
No LLM call or API spend occurred. This note closes the file drawer before the
held-out exact confirmation is run.

## Motivation

Rock Diagnosis is the information-only variant of RockSample introduced by
Araya-Lopez, Buffet, and Thomas, *Active Diagnosis Through Information-Lookahead
Planning* (2013). Rock types are static latent variables, moves are deterministic,
and a noisy long-range sensor becomes more reliable near a rock. The paper explicitly
identifies paths of initially uninformative moves that enable informative future
checks as its reason that lookahead is needed.

## Unregistered Screen

Before this tracked implementation existed, a temporary CPU-only prototype used the
paper's Figure 4 `5-7` rock layout, fixed left-centre entry `(0, 3)`, an eight-step
horizon, a uniform prior over all 32 rock-type vectors, the paper's exponentially
decaying sensor (implemented through `pomdp_py`'s RockSample observation model with
half-efficiency distance `log(2)`), and seed 1304. It selected actions by **incremental
EIG**, not accumulated negative entropy.

The screen used 500 paired trajectories at each K. Its width arm received the same
number of virtual candidate-proposal cells used by the depth-two root tree: one shared
root cell plus one current-state expansion for every feasible root outcome, with
deduplication. Values below are reductions in final posterior entropy; positive favors
depth two.

| K | d2 - shared d1 | 95% bootstrap CI | d2 - call-matched width | 95% bootstrap CI |
| ---: | ---: | --- | ---: | --- |
| 2 | +0.0317 | [+0.0189, +0.0456] | +0.0105 | [-0.0018, +0.0238] |
| 3 | +0.0881 | [+0.0701, +0.1067] | +0.0528 | [+0.0332, +0.0727] |
| 4 | +0.2271 | [+0.2003, +0.2539] | +0.1932 | [+0.1661, +0.2205] |

This identifies a plausible dynamic coupling mechanism, but the map, start state,
seed, and K values were viewed while developing the screen. They are not evidence for
the project claim and are not used to select a conclusion.

## Next Step

`ROCK_DIAGNOSIS_CONFIRMATION_PREREGISTRATION.md` freezes a held-out `3-6` Figure 4
map and new seed 2304 before its first execution. A subsequent 32-trajectory mechanics
smoke inadvertently used that source map and indices `0..31`; it passed its small-n
screen but is quarantined and cannot count as confirmation. The amendment freezes a
disjoint 2,000-trajectory confirmation on indices `32..2031` before it is run. Only
that result can gate a later bounded LLM candidate-proposal pilot.
