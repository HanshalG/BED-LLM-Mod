# ChemBench costed-repeat difficulty-sharding binding

Date: 2026-08-16

## Pre-result status

The serial `a078e7d5` execution was terminated before writing a result. Its only
output was nonnumeric phase progress: easy primary d3, d2, and d1 completed, then
the cost-blind d3 control continued exact recursion. No action, repeat count,
loss, comparison, condition, or gate value was emitted or inspected. A separate
bounded transition-cache benchmark was slower and was not adopted.

The sharding clarification was frozen before implementation:

- `CHEMBENCH_COSTED_REPEAT_CORRIDOR_DIFFICULTY_SHARDING_CLARIFICATION_20260816.md`
  SHA-256 `2b87d325e3265ba485046469da50b5e2de432904c43cb69bbeb8b94c018a4454`.

## Implementation

The runner now supports:

```text
--difficulty easy|medium|hard
--assemble-shards EASY MEDIUM HARD
```

Each shard runs the unchanged complete one-difficulty evaluator and atomically
writes its full slice with implementation, protocol-chain, source, settings, and
zero-call bindings. The assembler requires exact ordered coverage and rejects any
wrong schema, identity, commit, protocol, source, settings, calls, cost, or slice
difficulty before invoking unchanged `apply_gate` once.

Bound hashes before commit:

- runner: `cccd45c78006c397058fc7b22025991fd0792e8e73f6550e3b5ce3191a7927ff`
- focused test: `eac51c63e6b0930f8d90413965b6750ecbc2ad27a41ef7bb3340282c3bd45b61`

The focused suite passes `11/11`; `pytest -q tests/test_chembench*.py` passes
`111/111` in 149.33 seconds under concurrent macOS background load. Python
compilation and `git diff --check` pass.

No response value, model, network, cluster, or paid call was opened. Exactly one
easy, medium, and hard shard may run after this implementation is committed and
pushed; only their fail-closed assembled result has scientific authority.
