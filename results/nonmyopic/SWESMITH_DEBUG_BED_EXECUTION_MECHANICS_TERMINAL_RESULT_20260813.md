# SWE-smith Debug-BED execution mechanics terminal result

Date: 2026-08-13

Status: **infrastructure failed closed; scientific mechanics unmeasured**

## What passed before execution

The exact eight-task cohort reproduced from the pushed V2 source boundary.
Private aggregate parsing found six structurally eligible 2--4-hunk tasks and
two frozen structural invalids with 7 and 9 hunks. The eight tasks referenced
seven distinct released images. All seven images were available from the exact
pinned SWE-smith 0.0.4 Docker registry path and retagged under DebugGym's local
alias. Every image is `linux/amd64`; the local Docker host is `linux/arm64`.

## Terminal failure

The first structurally eligible task was launched in two fresh containers with
the exact released image under Docker's requested amd64 platform. Both arms
failed before `/bin/true` executed:

- exit codes: `[133, 133]`;
- stdout bytes: `[0, 0]`;
- stderr bytes: `[140, 140]`.

The raw stderr is not serialized because it is not needed for the frozen gate.
The failure stage is `container_process_start_before_bin_true`. The native
DebugGym reset, initial tests, PDB handshake, counterfactual hunk worlds, test
status matrix, exact planner, and patch endpoint never opened.

The protocol permits no architecture exclusions or migration after observing
the mechanics cohort. Therefore the exact Debug-BED mechanics construction is
closed. Remaining tasks were not executed because one eligible-task handshake
failure already makes the conjunctive gate impossible.

## Interpretation

This is an infrastructure null, not a non-myopic debugging result. The source
admission remains useful evidence that SWE-smith has 10,092 composed defects,
native debugger actions, and regression-protected endpoints. Nothing here
measures an adaptive dependency, depth-two value, myopic value, semantic belief
quality, or patch efficacy.

A future debugger experiment must be a genuinely new prospective protocol on a
native amd64 execution host, frozen before selecting or opening its cohort. It
cannot reuse this exact mechanics cohort as fresh evidence or describe this run
as a scientific failure.

## Privacy and accounting

No individual instance ID, repository, source path, problem statement, patch,
test name, test output, trace, gold fix, or endpoint is public. No opportunity,
development, confirmation, or reserve task was opened.

- OpenRouter calls: `0`
- OpenRouter cost: `$0`
- OATML cluster use: none

## Bindings

- mechanics protocol SHA-256:
  `6a786c696e7288ab54b2fe4b82602ff82d5d7ea85bdb79be6c630aeb6580b206`
- terminal result:
  `results/nonmyopic/swesmith_debug_bed_execution_mechanics/TERMINAL_RESULT.json`
- independent audit:
  `results/nonmyopic/swesmith_debug_bed_execution_mechanics/TERMINAL_AUDIT.json`
- audit implementation:
  `scripts/swesmith_debug_bed_mechanics_terminal_audit.py`

This result authorizes no model call or descendant.
