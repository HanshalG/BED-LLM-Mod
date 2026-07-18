# COPEx Direct-Proposal Thinking Quality Probe Preregistration

Frozen on 2026-07-18 after the completed non-thinking quadrature pilot and before
any thinking-model proposal is requested.

## Purpose

The completed quadrature pilot establishes a real grid depth-two advantage but fails
the LLM d2 versus d1/width gate. Its direct-angle logs are highly generic: 94.6% of
emitted angles are cardinal or diagonal defaults, with repeated three-angle cells
across distinct simulated child beliefs. This probe tests whether 26B Gemma reasoning
produces more useful, state-responsive direct proposal pools before a thinking d2
policy run is considered.

## Fixed Procedure

- Reconstruct the eight initial states from completed run
  `copex-direct-proposals-quadrature-pilot-20260718` (seed 46022), including its
  finite particle support and initial sensor location. No policy is rerun and no
  observations, selected actions, or endpoints are generated.
- On each state, request exactly the existing three-angle JSON cell from
  `google/gemma-4-26b-a4b-it` with `thinking: true`, a 2,048-token thinking limit,
  512-token final limit, temperature zero, and the same strict parser/boundary map.
- Score thinking, archived non-thinking LLM, and fixed-grid pools with the same
  complete-support three-node Gaussian-quadrature immediate EIG.
- Store raw cells, parser failures, exact scores, angle diversity, tokens, and cost.

## Decision Rule

This is a proposal-quality mechanism screen, not a policy result. A thinking d2 pilot
is warranted only if thinking has a strictly positive mean best-pool immediate-EIG
difference against non-thinking across the eight paired states **and** wins on at
least five of them, with no terminal parser failure. Otherwise the direct-thinking
route is rejected without a d2 rollout.
