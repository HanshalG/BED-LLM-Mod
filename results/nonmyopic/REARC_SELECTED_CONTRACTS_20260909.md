# Selected RE-ARC contracts and executable representation

Inspected only the four generators/verifiers selected in the frozen manifest from
upstream e5b7f1d06362a76f9d3b8c25154ff1fafca897ce. No task was executed and no
input/output examples were generated or read. Upstream license is MIT. Source
solutions are evaluator-only information and must never enter proposer prompts.

| ID | Generator structure | Output shape | Verifier steps |
|---|---|---|---:|
| 1a07d186 | Colored parallel lines, matching and distractor dots; optional transpose | Input shape | 19 |
| 1b2d62fb | Two colored panels separated by a bar; optional transpose | One panel, separator removed | 20 |
| 67e8384a | Connected colored pattern on a canvas | Twice input height and width | 4 |
| 264363fd | Template and bounded placement attempts for rectangular regions and markers | Input shape | 49 |

These are structural source observations, not model difficulty measurements. The
last generator bounds two placement loops but may place fewer objects than
requested. Runtime feasibility and validity rates remain unmeasured. The first
three also use randomized construction; do not silently resample difficult cases
until they pass a desired opportunity test.

The source verifiers are straight-line compositions, including higher-order DSL
values used later as callable operations. All four convert to the same graph
schema without special cases. The generic DSL contains160 function definitions
and28 literal constants; typing aliases are excluded from candidate constants.
Reference step counts are19/20/4/49, so an eight-operation handpicked language
would not represent this entire source panel.

## Implemented representation check

scripts/rearc_program_graph.py accepts a bounded graph of canonical x0..xN steps,
named callable operations and backward references, with an explicit final output.
It rejects imports, attribute access, unbound calls, forward references and loops.
Maximum128steps is an engineering cap; all selected references fit. This is NOT a
runtime sandbox or static type checker: intermediate values may not actually be
callable, primitives may receive wrong types, and higher-order operations may
expand work. Validating a graph does not prove safe or successful execution.

Six tests pass in .08s, including higher-order composition and rejection of unsafe
source constructs without execution. All four selected references pass conversion.
The original source remains untouched. We have not supplied any reference graph
to an LLM or claimed a discovery result from compiling it.

## Next dependency

Retain all four tasks. Build bounded isolated execution for graphs using the full
generic DSL, with time/memory/output limits and an environment without credentials.
Use the same runtime for symbolic and LLM proposals. Static graph validation is
only the entry filter. Freeze source-generation seeds, attempt limits, public-input
pool and evaluation handling before source runtime checks; keep output-dependent
metadata private. Different output sizes require an explicit shared predictive
outcome space, not silently comparing same-size survivors.

After runtime/source checks, a small prospective Luna-medium proposal qualification
must compare context/history-aware proposals with equal-call blind proposals and
a productive symbolic baseline. Whole-grid observations may saturate after one
example, especially for the simple mirror rule; no task may be dropped to create
a depth curve. Planning opportunity and independent prediction remain unproven.

Previous turn was source progress; current turn audits all four selected contracts
and implements a shared representation without excluding the hardest program.
Cost$0; balance23.693468061 and conservative daily remaining4.11174654 unchanged.
Goal active/unachieved; no paid experiment authorization, cluster or automation.
