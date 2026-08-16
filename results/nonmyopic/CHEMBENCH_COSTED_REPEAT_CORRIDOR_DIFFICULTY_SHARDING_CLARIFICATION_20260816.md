# ChemBench costed-repeat difficulty-sharding clarification

Date frozen: 2026-08-16

## Pre-result status

The corrected serial execution from pushed commit `a078e7d5` was terminated before
writing any result or numerical metric. It emitted only phase markers showing that
easy d3, d2, and d1 had completed and that the easy cost-blind d3 control was still
running. No root action, repeat count, loss, comparison, condition, or gate value
was emitted or inspected.

The cost-blind control is intentionally much larger than the cost-aware policies:
it plans as though all repeat allocations cost one decision while execution still
charges their real well cost. Process profiling confirms exact dynamic-program
recursion, not an error or stalled process. A bounded transition-cache benchmark
was slower and is rejected.

## Independent difficulty shards

The three frozen difficulties are conditionally independent once the pinned
source, implementation, protocol hashes, query seed, CRN seed, and global settings
are fixed. The runner may therefore evaluate `easy`, `medium`, and `hard` in
separate processes, including concurrently. Each process executes the exact same
`make_costed_bank`, dynamic suite, fixed-support control, full-support control,
immutable proposal replay, and CRN transcript replay already bound by the
evaluator-efficiency amendment.

Each atomic shard must contain:

- schema and difficulty identity;
- the pushed implementation commit;
- exact hashes of every governing protocol/amendment;
- exact source binding and a canonical source-binding hash;
- a canonical hash of all frozen global settings;
- the complete one-difficulty result; and
- zero model, network, and dollar cost declarations.

## Fail-closed assembly

The final assembler accepts exactly one shard for each ordered difficulty
`easy`, `medium`, and `hard`. It must reject a missing, duplicate, malformed,
wrong-order, wrong-commit, wrong-protocol, wrong-source, wrong-settings, or
nonzero-call shard. It then extracts the three unchanged slice payloads and calls
the original `apply_gate` exactly once. The assembled result binds each shard's
SHA-256.

Partial shards and phase logs have no scientific authority. A shard may be reused
only by the exact pushed implementation and assembler binding that created it.
No cohort, response, policy, control, seed, scenario, action, budget, loss, or gate
changes. This clarification authorizes no model, network, cluster, or paid call.
