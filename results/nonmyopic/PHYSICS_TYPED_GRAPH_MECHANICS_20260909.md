# Structured physics proposals: tested compiler, untested model interface

The previous paid repair run failed on unmatched parentheses before scoring.
This change addresses that instrument failure, not the scientific first-link gap.
No banked response is corrected, partially accepted, rescored or regenerated.

New scalar_graph.py accepts one to eight topologically ordered graphs, each at
most64nodes. Explicit numeric constants, variable indices and arithmetic/function
operations compile deterministically into the existing bounded safe interpreter.
Strict keys, arity, earlier-node references, finite constants, dimensions and
null unused fields are enforced. Cycles and forward references fail. Repeated
subexpressions may share nodes; expansion is checked at every node against the
existing byte/node/depth limits. No eval, exec, arbitrary code or imports.

The outer JSON schema alone does not enforce reference topology or arity. Models
can still return invalid graphs. Accepted graphs avoid parenthesis-balance errors,
but domain errors, wrong mechanisms, numerical overflow, poor coverage and wrong
predictive ranking remain possible. Bounded representation may itself exclude
complex useful formulas; it is not a complete physics language.

New physics_graph_proposals.py retains exactly the existing public history,
semantic context, initial diagnostics, guards and scale prior. It replaces the
system output contract with graph instructions, without contradictory string-array
instructions. No target labels or true formula enter this message builder.

Verification:35tests in .59s, lint passes. Includes100deterministic random arithmetic
graphs against independently computed values, shared trigonometric subexpressions,
topology/type/resource adversaries, whole-response rejection, syntax deduplication,
unrepaired domain failures and no fourth-observation leakage into the control.
These are zero-call mechanical tests, not an LLM schema/semantic gate.

Next: prospectively bind a separate two-call paired development runner with this
schema/decoder/prompt, unchanged history routing, exact numerical updater, fixed
seed and full worst-case budget. Test receipts/sealing/failure replay before any
request. Any repair signal still requires a fresh-cohort, matched-compute test;
do not promote the selected opened case to non-myopic evidence. Eventual depth
comparison additionally requires joint answer/updater prediction and genuine
planning headroom. A graph syntax pass alone authorizes none of those claims.

Live account unchanged245/221.234591189/23.765408811; conservative Sept9remaining
4.18368729. No paid calls, cluster, automation or protected-runtime changes.
Previous turn was progress (measured paid instrument failure); current turn adds
tested compiler and interface, leaving the research goal unachieved.
