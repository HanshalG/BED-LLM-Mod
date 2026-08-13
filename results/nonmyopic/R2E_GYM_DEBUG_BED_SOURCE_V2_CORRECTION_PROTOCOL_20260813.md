# R2E-Gym Debug-BED source V2 correction protocol

Date frozen: 2026-08-13 Europe/London

Status: **prospective audit-code correction; zero model calls**

## V1 disposition

The exact V1 source audit is terminally `source_failed_closed`. Every immutable
source, population, schema, eligibility-size, repository-diversity, split, and
privacy gate passed, but `native_debugger_contract` failed because the checker
searched the pinned DebugGym file for the nonexistent literal `class Pdb`.
Direct code inspection shows the released registered implementation is named
`PDBTool` and exposes PDB start, restart, persistent breakpoint, and command
interfaces. No task payload or endpoint was opened.

## Sole correction

V2 changes only the debugger symbol assertion from `class Pdb` to
`class PDBTool`. It keeps V1's exact:

- repository, dataset, shard, and runtime-file bindings;
- 4,578-row train population and 14-field schema;
- six-column metadata projection and forbidden payload columns;
- eligibility thresholds and uniqueness requirements;
- salt and ordered 16/64/64/96/reserve split;
- minimum 256 eligible rows across eight repositories;
- public serialization boundary, downstream mechanics gates, and accounting.

V2 must reproduce the exact V1 manifest's population counts, selection counts,
split hashes, and source/runtime hashes. It must also prove that V1 failed only
the malformed native contract gate and that the pinned DebugGym PDB file
contains `class PDBTool`, `start_pdb`, `restart_pdb`, and `interact_with_pdb`.

No threshold, subset, salt, ordering, cohort, source, or scientific gate may
change in response to V1. A V2 pass authorizes only the same separately frozen,
predicate-pushed 16-row structural screen specified by V1. It authorizes no
task execution, model serving, endpoint, or efficacy claim.

## Accounting

- OpenRouter calls: `0`
- OpenRouter cost: `$0`
- OATML cluster use: none

