# SWE-smith Debug-BED V3 structural privacy terminal result

Date: 2026-08-13

Status: **privacy-ordering failed closed before execution**

## Failure

The pushed V3 metadata selector is valid: it projected only IDs, image names,
and F2P/P2P list lengths to define a fresh twelve-row screening prefix from
the unopened V2 reserve.

The subsequent structural-screen implementation violated the frozen ordering.
Before filtering to those twelve IDs, it called `pyarrow.parquet.read_table`
without a column projection on all eleven shards. That materialized full rows,
including patches, problem statements, and test values, for all 50,137 tasks.
Only twelve rows were retained in the private file and no private value was
serialized publicly, but the protocol forbade opening any nonselected payload.

Therefore the exact V3 structural result and its eight-slot hash authorize
nothing. The unlaunched mechanics protocol/workflow was deleted before push;
no task image execution, status matrix, PDB transcript, planner score, model
call, patch endpoint, opportunity, development, or confirmation outcome
opened.

## Interpretation

This is a privacy/order-of-operations null, not evidence about Debug-BED's
horizon opportunity. The passing native-amd64 venue attestation and V3
metadata-only source admission remain valid. A successor requires a fresh salt
and cohort excluding all twelve V3 screening IDs, with predicate-pushed exact
row retrieval that reads no nonselected payload.

## Accounting

- OpenRouter calls: `0`
- OpenRouter cost: `$0`
- OATML cluster use: none
