# Rock Diagnosis Exact Confirmation Preregistration

Registered: 2026-07-15, before the first `3-6` execution. See
`ROCK_DIAGNOSIS_CONFIRMATION_AMENDMENT.md` for the quarantined mechanics smoke and
the final held-out trajectory index range.

## Question

Does depth-two incremental EIG beat both a shared-candidate one-step policy and a
candidate-call-matched, wider one-step policy when future sensor quality depends on
the rover's current position?

This is an exact, zero-LLM-call mechanism confirmation. It does not itself establish
an LLM result; it gates a separate bounded candidate-proposal pilot.

## External Domain And Frozen Adaptation

- Source: Araya-Lopez, Buffet, and Thomas (2013), *Active Diagnosis Through
  Information-Lookahead Planning*, Figure 4 `3-6` map, page 10:
  `https://members.loria.fr/olivier.buffet/papiers/jfpda13-b.pdf`.
- Frozen grid and rocks: side length 6; rock positions `(4, 1)`, `(1, 4)`, and
  `(4, 4)` transcribed from Figure 4.
- The paper's figure specifies rock layout but not an entry position. This adaptation
  fixes the conventional left-centre start `(0, 3)` before execution.
- Latent target: the full 3-bit good/bad rock-type vector, with uniform prior over its
  eight configurations. It does not change over time.
- Acquisition actions: legal cardinal moves and `check-i` sensor queries only.
  Sampling and exiting are removed, as required by Rock Diagnosis.
- Transition/observation core: `pomdp_py==1.3.5.1` RockSample transition and sensor
  models. Motion is deterministic; check accuracy is the library's distance-dependent
  model with half-efficiency distance `log(2)`, matching the paper's exponential
  form. The posterior is exact enumeration over the eight target vectors.
- Decode: at each round and at the endpoint, choose the MAP vector. Decode is never an
  EIG candidate.

## Frozen Evaluation

- Seed: 2304.
- Paired trajectories: 2,000.
- Horizon: 8 actions.
- Candidate widths: K in `{2, 3, 4}`.
- Bootstrap: 10,000 deterministic paired percentile resamples.
- Common random numbers: every arm shares each trial's hidden vector; observation
  uniforms are keyed by trial, position, checked rock, and repeat count. Candidate
  pools are history-keyed and deterministic.

For every current state, a candidate cell is a deterministic random ordering of legal
actions truncated at K. The shared depth-one and depth-two arms receive exactly the
same base cell. The depth-two value is

`EIG(a) + sum_o p(o | a, b) max_{a' in C(b', o)} EIG(a')`.

The one-step width control gets the same base cell plus one independent current-state
candidate cell for every nonzero-probability root-outcome cell that depth two scores.
Cells are unioned and deduplicated. Thus the control makes the same number of
candidate-proposal calls as the full depth-two root tree but uses all of them to widen
the current action set. Candidate orders are nested within a cell as K grows.

## Outcomes And Decision

Primary outcome: paired reduction in final exact posterior entropy. Secondary outcomes:
full-vector MAP accuracy, true-vector log posterior, root movement rate, selected EIG,
and complete action traces. Positive entropy reductions favor depth two. The tracked
JSON keeps all trial-level final metrics; complete per-step traces are written as a
locally ignored, reproducible `TRACES.jsonl.gz` artifact.

The exact confirmation passes if at least one K has both:

1. depth two minus shared depth one final-entropy reduction strictly positive with a
   paired 95% bootstrap CI excluding zero; and
2. depth two minus candidate-call-matched width final-entropy reduction strictly
   positive with a paired 95% bootstrap CI excluding zero.

Regardless of the result, publish the full K table and all mechanics checks. A pass
permits one separate <=10-task, <=$1 LLM candidate-proposal pilot; a fail is recorded
as a non-promotion and redirects exploration without suppressing the result.
