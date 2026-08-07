# RegretBench Distinct Semantic Action Amendment

Date: 2026-08-07

Status: frozen before any RegretBench support or policy model response.

## Problem

The policy prompt prohibits repeating an answered clarification, but the
original mechanics required only that each question map to a supported official
facet. In RegretBench, the environment's answer is a deterministic slot value
for that facet. Asking the same facet again therefore supplies no new evidence.
An answer-conditioned LLM refresh could nevertheless assign a different belief
after the repetition and create artificial terminal improvement.

## Amendment

Two executed clarification actions are distinct exactly when both map to
supported official facets and their mapped facet names differ.

- The enriched exact-10 smoke now requires all three branch-selected second
  actions to be distinct from their corresponding first actions.
- Each primary formal policy must have at least `40/64` supported second
  actions whose facet differs from its first action.
- The Luna naive exact-10 smoke requires all four second actions to be distinct.
- The optional formal Luna baseline uses the same `40/64` novelty target as an
  availability diagnostic only. Its result still cannot affect primary status.
- The independent zero-call verifier reconstructs action novelty from the
  official mapper and fails on any disagreement with the public result.

No prompt, task, seed, request count, endpoint, scientific threshold, or budget
changes. This amendment can only reject repeated-action trajectories; it cannot
rescue or improve an outcome.
