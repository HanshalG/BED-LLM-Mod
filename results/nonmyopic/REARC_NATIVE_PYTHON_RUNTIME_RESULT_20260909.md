# Native Python representation: isolated mechanics

Implemented a separate native list-grid transform(grid) executor. Host-side code
only parses/validates AST; candidate execution is confined to a fresh pinned
container for each program/input. The mount contains only executor/validator/grid
checker, not benchmark source, other examples, hidden outputs or credentials.
Nonroot/read-only/no-network,256MiB,2CPU seconds,15seconds host wall limit,
32PIDs, output-size bound and named cleanup remain explicit.

Interface allows ordinary functions/helpers, loops, mutable local collections,
comprehensions and a fixed math/collections/itertools/functools/heapq library set.
File/process/dynamic-evaluation builtins are not supplied. Import/private-name
checks are defense in depth, not a claim that Python AST filtering is a security
boundary. The OS/container boundary remains necessary. This is a constrained
standard-Python interface, not unrestricted Python or a full Poetiq environment.

Seven unit tests pass: module/private-name/signature restrictions, loops/helpers,
no host candidate execution, container limits and cleanup on timeout.
Nine actual synthetic container cases passed:
- Rectangular transpose with output resizing.
- Flood fill for an enclosed region and for a border-connected region, preserving
  wall cells in both cases.
- Two executions of a stateful counter in separate containers both return1.
- Invalid output color is rejected by the grid validator.
- Attempted file access fails with NameError because open is unavailable.
- Infinite loop and excessive allocation terminate with exit137. This records
  termination, not independent attribution of the kernel signal's exact cause.

All examples are handcrafted, not benchmark inputs or repaired old outcomes.
No containers remain, no model calls or API cost. Code/request/response artifacts
are banked in rearc_python_runtime_smoke_20260909 with implementation hashes.

## Remaining readiness work

This is not complete security certification or a paid-qualified representation
experiment. Before calls, test Python-version/annotation compatibility, permitted
library imports and malformed code, deterministic repeat behavior beyond the
state-counter fixture, and response-budget failures. Pin the exact interface in
model prompts, connect native programs to the existing fixed-slot predictive
scorer, and test the actual shared-plan native-vsDSL controller and replay.
No performance-based task filtering, first-fit stopping or discarded prediction
failure mass may be introduced. An isolated executor does not establish predictive
support, branch fidelity, horizon opportunity or non-myopic efficacy.

Previous goal turn: progress by public runtime/source reassessment. Current:
progress by implemented/tested isolated native execution. Account unchanged,
conservative London Sep9 remaining3.65706419. No cluster or automation changes.
Goal remains unachieved.
