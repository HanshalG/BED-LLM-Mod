# RE-ARC candidate: active example acquisition, not unrestricted query synthesis

Pinned [upstream source](https://github.com/michaelhodel/re-arc/tree/e5b7f1d06362a76f9d3b8c25154ff1fafca897ce)
at e5b7f1d06362a76f9d3b8c25154ff1fafca897ce in a no-checkout partial clone.
The upstream main.py was read as text, not imported. No task was executed, no
generated grids displayed, and no task-specific solver bodies inspected.

The dataset builder selects a task-specific generator and verifier, generates an
input/output pair, verifies the output, rejects input==output examples, and dedupes
by the input hash. Its difficulty summary includes output-derived properties.
Those output-dependent acceptance/ranking decisions cannot silently become a new
policy's free oracle. The stock while-loop also has no fixed attempt ceiling.

These facts favor a carefully declared pool-based label-acquisition study over
claiming that every arbitrary grid is a valid experimental intervention. A public
input pool can depend on a task's distribution, but all policies must see the
same pool and their initial beliefs must condition on it. Do not treat it as
uninformative or reveal task IDs, verifier code, generated outputs, difficulty
scores using outputs, or reference-solver success indicators to the policy.

An eventual endpoint would be prediction on separate grids, with a fixed proper
score and target set across depths. Rich whole-grid answers may resolve the rule
after a single query, so planning opportunity is unproven. Familiar ARC training
tasks also create memorization risk. Neither source availability nor plausible
LLM relevance establishes a non-myopic result.

## Frozen source-inspection scope

Metadata-only AST parsing found 400 verifier IDs. Before inspecting task-specific
content, select the four smallest SHA256('bed-rearc-source-v1:'+id):
1a07d186, 1b2d62fb, 67e8384a, 264363fd. Keep all four even if incompatible or easy;
no replacement. The manifest binds the full inventory and verifier file hash.
Parsing loads source bytes but does not constitute cryptographically sealed
ingestion; unselected bodies remain undisplayed and unused in selection.

Next inspect only these generators/verifiers for valid input domains, output
shapes, executable DSL requirements and generator rejection behavior. Then decide
whether a safe typed proposer and source-only opportunity test are feasible.
Do not download or run the full generated dataset or expose reference solutions
to an LLM. Qualification must include equal-call blind proposals and a productive
symbolic solver; a positive over random alone would not suffice.

One test confirms selection depends on function names and does not execute source.
This is a prospective source scope, NOT paid authorization or a claim of external
predictive calibration. It does not reopen the closed Number Game interfaces.

Previous turn was implementation progress; current turn establishes a new pinned
source contract and immutable four-task inspection scope. Cost $0, account usage
221.306531939 and balance23.693468061 unchanged; daily remaining4.11174654.
Goal active/unachieved. No cluster, automation, or protected runtime changes.
