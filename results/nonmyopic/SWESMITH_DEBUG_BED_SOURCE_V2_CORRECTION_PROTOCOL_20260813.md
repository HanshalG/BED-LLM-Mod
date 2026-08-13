# SWE-smith Debug-BED source V2 correction protocol

Date frozen: 2026-08-13

Status: **prospective metadata-only correction; zero model calls**

## Scope

V1 remains failed because it required the raw `train-789` and `test-125` lists
to be disjoint from DebugGym's published exclusions. Source inspection after
V1 showed that the released loader instead passes both named lists and the
exclusions to `filter_problems`, which removes excluded IDs when materializing
either split.

V2 changes only that metadata interpretation. It defines:

- effective development = published `train-789` minus published `excluded`;
- effective confirmation = published `test-125` minus published `excluded`.

The raw named lists must still contain exactly 789 and 125 unique mutually
disjoint population IDs. The effective lists must contain exactly 760 and 121
IDs, remain disjoint, and reproduce the released filtering code. Excluded IDs
are unavailable to every V2 split. All immutable source bindings, composition
thresholds, privacy constraints, mechanics/opportunity selection, scientific
gates, controls, endpoint sealing, and accounting from V1 remain unchanged.

No individual ID, task payload, patch, test name, source file, execution output,
or endpoint was inspected before this correction was frozen. V2 is a distinct
source admission, not a retroactive V1 pass.

## Exact V2 split

Composed IDs outside effective development, effective confirmation, and all
exclusions retain the unchanged V1 hash ordering and allocations:

| Split | Count |
|---|---:|
| mechanics | 8 |
| opportunity | 64 |
| development | 760 |
| confirmation | 121 |
| reserve | all remaining eligible composed tasks |

Every V1 source and execution-mechanics requirement remains binding. V2 failure
closes this source line; there is no further split repair.

## Accounting

- OpenRouter calls: `0`
- OpenRouter cost: `$0`
- OATML cluster use: none
