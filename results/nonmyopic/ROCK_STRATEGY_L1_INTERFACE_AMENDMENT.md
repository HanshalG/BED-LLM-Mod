# Rock Strategy-Prior L1 Interface Amendment

Recorded on 2026-07-16 after the preregistered interface-only smoke and before the
formal L1 run. No policy endpoint, arm comparison, selected action, or strategy trace
from the smoke was read.

## Permitted Observations

The smoke completed with zero terminal failures and legal selected actions. It used 10
physical API attempts: 7,538 prompt tokens, 3,149 completion tokens, zero reasoning
tokens, and `$0.00206105`. Two raw responses failed strict schema validation and both
validated after the single registered feedback retry:

1. An `ever_observed` predicate used `G`/`B` instead of lowercase `good`/`bad`.
2. An `at_rock` predicate nested the predicate name instead of placing `at_rock` in
   the required `kind` field.

All root-mix, action-legality, shared-cell, width-call, exact-scorer-unit, random-cell,
and zero-LLM rollout-scoring mechanics passed.

## Interface-Only Changes

- The strategy prompt now explicitly states that observation outcomes are lowercase
  `good` and `bad` and illustrates the canonical
  `{"kind":"at_rock","rock_id":0}` predicate shape.
- Physical request accounting now includes rejected attempts as well as accepted
  cells; accepted cells remain reported separately.
- Independent paired trials may execute concurrently, with frozen formal concurrency
  32. This changes wall-clock throughput only. Every trial retains its keyed truth and
  observation seeds, its own five policy states, and the same exact scorer.

The maps, fresh formal seed 12032, arms, K=4, horizon 2, eight rounds, 30 paired
trajectories per map, model, non-thinking mode, temperature, grammar, retry limit,
exact dynamics, endpoints, bootstrap, gate, and `$1.00` hard run cap are unchanged.
The measured linear cost projection is approximately `$0.25`; the more conservative
frozen projection remains `$0.80`.
