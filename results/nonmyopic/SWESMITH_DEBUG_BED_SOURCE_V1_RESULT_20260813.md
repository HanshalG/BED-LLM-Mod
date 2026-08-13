# SWE-smith Debug-BED source V1 result

Date: 2026-08-13

Status: **source failed closed; zero model calls**

## Result

The metadata-only V1 audit establishes a strong released substrate:

- 50,137 executable SWE-smith tasks across 128 repositories and 128 images;
- 5,865 `combine_file` and 4,227 `combine_module` composed defects;
- every composed task has at least one fail-to-pass and one pass-to-pass test;
- exact DebugGym code/data provenance and eleven content-addressed Parquet
  shards;
- native persistent PDB, official evaluation, regression-protected scoring,
  upstream-remote removal, and seedable environment contracts.

V1 nevertheless fails its conjunctive official-split gate. The published
DebugGym config contains 789 development IDs, 125 confirmation IDs, and 37
excluded IDs. The named lists are unique and mutually disjoint, but the
excluded list overlaps 29 development and 4 confirmation IDs. V1 incorrectly
required the raw named lists themselves to be disjoint from exclusions, while
the released loader implements named splits by filtering exclusions at load
time.

V1 remains failed and is not repaired or relabeled. No task payload, problem
statement, patch, test name, repository source, execution output, gold fix, or
endpoint was opened. A distinct V2 protocol may bind the released effective
split semantics because this mismatch was found using metadata only and before
any scientific observation. V2 cannot alter any scientific threshold, split
ordering, or privacy boundary.

## Accounting

- OpenRouter calls: `0`
- OpenRouter cost: `$0`
- OATML cluster use: none

This V1 result authorizes nothing.
