# Rock Diagnosis LLM Candidate-Proposal Pilot Registration

Registered: 2026-07-15, after the independent exact `3-6` confirmation and before
the first provider request. The failed-closed first interface attempt and its bounded
retry repair are recorded in `ROCK_DIAGNOSIS_LLM_PILOT_RETRY_AMENDMENT.md`.

## Scope

This is one bounded **exploration** pilot, not a confirmatory experiment. It tests
whether an LLM-restricted candidate proposer preserves the already-confirmed
depth-two-over-width mechanism in the external Rock Diagnosis task. It may use at
most eight paired trajectories and has a projected OpenRouter cost of `$0.15`, a hard
run cap of `$0.50`, and no reasoning tokens.

## Frozen Design

- Environment: Figure 4 `3-6` Rock Diagnosis layout from Araya-Lopez, Buffet, and
  Thomas (2013), fixed entry `(0, 3)`, with the exact `pomdp_py==1.3.5.1` transition
  and distance-dependent check likelihood used in the preceding confirmation.
- Latent target: the full static three-bit rock-type vector. The exact posterior
  enumerates all eight configurations; endpoint decode is full-vector MAP.
- Seed: `2304`; 8 paired trajectories; 8 action rounds; candidate width `K=3`.
- Candidate proposer: `google/gemma-4-26b-a4b-it` through OpenRouter with
  `thinking: false`, temperature `0`, maximum 128 output tokens, and one bounded
  retry. The prompt lists only the current legal action IDs, exact posterior summary,
  position, map, and realized history. It does not reveal the hidden truth.
- Candidate parser: exactly `{"action_ids":[...]}` (a complete fenced JSON object is
  accepted); exactly three distinct IDs must be legal at the current position. No
  candidate is padded, repaired, or silently substituted. Exhausting the retry budget
  fails the pilot closed.
- Exact acquisition and controls:
  - `d1_shared`: maximum immediate exact EIG from a root candidate cell.
  - `d2`: the same root cell, maximizing exact
    `EIG(a) + sum_o p(o|a,b) max_{a' in C(b',o)} EIG(a')`.
  - `d1_call_matched_width`: the same root cell plus one independently proposed
    **current-state** cell for each nonzero root-outcome cell evaluated by `d2`;
    cells are deduplicated and scored only by immediate exact EIG.
- Common random numbers: all arms share the hidden vector. Sensor uniforms are keyed
  by `(seed, trial, position, checked-rock, repeat-count)`, so identical contextual
  checks share observations. Root candidate cells are cached and must be identical.

## Read Criterion And Promotion

Primary exploratory readout: paired reduction in final exact posterior entropy. A
positive `d2 - control` value favors depth two. Secondary readouts are entropy AUC,
full-vector MAP accuracy, true-vector log posterior, root movement, selected EIG,
candidate-cell sizes, token/cost usage, raw rejected parser attempts, and full action
traces.

The pilot is directionally promotable only when `d2` has a strictly positive mean
paired final-entropy reduction against **both** shared `d1` and call-matched width,
and the shared-root, legal-action, and candidate-call-accounting checks pass. The
eight-task intervals are descriptive only. A promotable outcome earns exactly one
separate paired, pre-registered confirmation whose trial count is powered from this
effect; any other outcome is ledgered as a non-promotion.

## Cost Projection

The worst-case logical candidate-call count is bounded by 8 trajectories × 8 rounds
× (1 root + 6 root-outcome continuations) for each of depth two and width, plus 64
one-step calls: at most 960 logical cells before caching. At 128 output tokens and
the concise structured prompt, `$0.15` is a deliberately conservative projected cost;
the OpenRouter adapter enforces the `$0.50` hard run cap before a charge can exceed it.
No LLM request is made by the dry-run mechanics test.
