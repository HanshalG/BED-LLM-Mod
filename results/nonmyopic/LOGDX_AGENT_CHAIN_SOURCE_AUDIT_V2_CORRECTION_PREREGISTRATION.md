# LogDx-CI Agent-Chain Audit V2 Correction Preregistration

Date: 2026-07-29

Status: **frozen before the corrected aggregate is computed**.

## Correction

The V1 audit reconstructs tool observations from the released deterministic
LogDx agent tools. A `grep` observation begins with a header that repeats the
query pattern. V1 excludes literals already present in the initial context,
but it does not exclude literals already supplied in an earlier tool call.
Consequently, repeating a previous query pattern can be misclassified as an
observation-dependent later action even though the literal was echoed by the
tool header rather than learned from the log.

This is an implementation error relative to the preregistered scientific
definition of a literal **learned from a prior observation**. The public V1
audit SHA-256
`f7f7289f33434a7e9b3a69bc7dd54abbaedbd55a92fb28be8e29090500c1d9ab`
is preserved and superseded, not overwritten.

## Frozen V2 Rule

V2 keeps the exact pinned source, `420` matched rows, case aggregation,
thresholds, score definitions, and all source/leakage gates from V1.

It changes only dependency detection:

- a later line number must appear as a line prefix in a prior observation,
  be absent from the initial context, and be absent from all prior tool
  arguments;
- a later search literal must appear in a prior observation, be absent from
  the initial context, and be absent from all prior tool arguments.

Prior arguments are compared through canonical JSON. This rejects query-header
echoes and repeated user-supplied line coordinates while retaining literals
that first occur in returned log content.

All original thresholds remain conjunctive:

- at least `25` matched distinct cases;
- at least `12` tool-use cases;
- at least `8` multi-tool cases;
- at least `6` corrected dependency cases;
- at least `5` corrected dependency cases with mean paired gain at least
  `.10`;
- positive all-case mean gain;
- corrected dependency-case mean gain at least `.10`;
- at least two corrected dependency types.

Failure closes LogDx before serving or policy work. Pass authorizes only the
same separately frozen ten-call serving test described by V1.

No model call, prompt change, case removal, threshold repair, or OpenRouter
spend is allowed during this correction.
