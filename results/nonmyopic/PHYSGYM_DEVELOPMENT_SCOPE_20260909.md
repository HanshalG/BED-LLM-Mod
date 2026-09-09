# Prospective PhysGym development source scope

At pinned fe68079c0921029dde679ed44bc3192dd3b270ab, parse full_samples.json as
data only. Select four development IDs by ascending SHA256('physgym-dev-v1:'+id),
before displaying context, expressions, solutions or code. Select by IDs alone;
do not replace an invalid, unsafe, high-dimensional or difficult selected task.
The JSON contains all source solutions in downloaded bytes; parsing it is not
sealed-data ingestion. Only the four selected task objects may be inspected or
emitted. Remaining objects stay undisplayed and unused in prediction or selection.

This permits source inspection, not paid calls. Record the file hash, full ID
inventory and selected IDs before emitting selected records. Inspect code as text,
never import or execute the benchmark module or generated source. Check declared
input ranges, domain assertions, stochasticity, dependencies and context leakage.
Require a safe supported expression or an isolated reviewed evaluator plus a
prospective input distribution before model proposals. Do not expose answer,
solution, equation, implementation, task ID or benchmark name to the eventual LLM.
Public context must be reviewed for directly giving away the answer; any semantic
versus blind comparison must report such leakage rather than count recall as
experimental discovery. No old failed interface is reopened by this source audit.
