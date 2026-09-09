# Public grid-to-program proposal interface

Implemented a source-independent prompt builder and strict four-graph response
parser. The prompt receives public input grids, explicitly observed index/output
pairs, and generic DSL signatures/docstrings/constants. It does not receive task
IDs, generator/verifier implementations, explanations derived from source solvers,
or held-out outputs. The complete generic documentation occupies13,310serialized
prompt bytes in a one-cell mechanics fixture; no task-specific grammar reduction.

Four graphs, each1..128steps, are required. A malformed graph rejects the complete
response; no partial-success filtering or automatic repair. Duplicate JSON keys
are rejected. Syntactically duplicate valid graphs are retained at parsing, so
future uniqueness diagnostics and prior choices cannot be hidden in the parser.
The system requests diversity; actual duplicate rates remain an empirical gate.
Provider schema compatibility is NOT yet measured by these local tests.

Public inputs are the same for history-aware and blind arms. Only observed output
pairs differ. A runner must construct each observation list from the prospectively
visible prefix and verify blinded-prompt invariance to withheld answers. Every
proposed pool is later evaluated/fitted against the same full demonstration history.
The builder never reads a benchmark or sample file and has no hidden-target argument.

Three tests pass in .09s: output visibility and no implementation bodies in generic
documentation, exact response count, complete-response invalidation, duplicate-key
rejection and strict schema metadata. No model call or benchmark example generation.

## Remaining before paid qualification

Freeze example seeds/splits and reveal order; implement a productive symbolic
baseline; fix posterior or forecast weights including duplicates and invalid cases;
then bind paired prediction thresholds and exact Luna-medium request/cost caps.
The requested model remains Luna-medium, but this interface adds no transport or
spending authorization. Re-read live catalog pricing at dispatch and reserve the
full token exposure, not a typical-call estimate. The 128KiB prompt/response byte
caps here are engineering bounds, not provider token budgets.

Successful schema parsing will not establish useful discovery, and successful
proposal prediction will not establish non-myopia. Preserve that distinction when
moving to the actual semantic gate and subsequent branch-fidelity experiment.

Previous turn was scoring progress; current turn implements the actual public
proposal interface without reference-task leakage. Cost$0, balance23.693468061,
daily conservative remaining4.11174654 unchanged. Goal active/unachieved.
