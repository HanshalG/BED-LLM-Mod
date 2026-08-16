# ChemBench costed-repeat evaluator-efficiency binding

Date: 2026-08-16

## Pre-result status

The pushed v1 runner was terminated with exit status 143 after 13 hours and before
writing `RESULT.json`. No root action, policy loss, comparison, condition, or gate
value was emitted or inspected. A macOS process sample showed the runner inside
the standard-library JSON encoder while its resident memory grew beyond 1.7 GB.
Code inspection identified two non-scientific causes: materialization of the full
proposal-cache JSON string and a second complete optimization for immutable
replay.

The evaluator-efficiency amendment was frozen before this implementation:

- `CHEMBENCH_COSTED_REPEAT_CORRIDOR_EVALUATOR_EFFICIENCY_AMENDMENT_20260816.md`
  SHA-256 `102ce68e4ca4316d2650bc8096356baf3defc6580735f5d24cb165c32ccec783`.

It leaves every scientific setting and prospective threshold unchanged.

## Implementation

- proposal records expose a read-only zero-copy view;
- canonical SHA-256 digests stream the exact sorted compact JSON byte sequence;
- d3, d2, and d1 share one planner's existing pure policy/proposal caches;
- every exact planner records only policy decisions reached by frozen CRN
  execution;
- a fresh immutable proposal cache re-executes all CRN trajectories from that
  transcript and requires exact actions, outcomes, losses, audit fields, and
  digests;
- proposal-record replay checks the complete immutable bank; and
- redundant full policy optimization during replay is removed.

Bound file hashes before commit:

- `environments/chembench_mopen/mechanics.py`:
  `247e875a4e61fe0c8498c3b88e5c93365cbeb053d180f868423131e03d7bef31`
- `environments/chembench_mopen/costed.py`:
  `4b23e70e605c2815cf8cb6dda1f98f4b02eeda69b557fffb187cf16a95630c57`
- `scripts/chembench_costed_repeat_corridor.py`:
  `b348307bdd28815f908f431baaa860c7c3ea1284bee795bebfec49442cf1d549`
- `tests/test_chembench_costed_repeat_corridor.py`:
  `65ab0996aaf53de501a74bdb8fb7d5309bf3c2ba617929381ccbdcf5b26d4a3c`

## Verification

Focused tests cover:

- streaming versus materialized canonical digest equality;
- shared-instance versus isolated-instance d3/d2/d1 output equality on a complete
  small bank; and
- exact immutable transcript replay of all CRN trajectories.

`pytest -q tests/test_chembench*.py` passes `110/110` in 56.46 seconds. Python
compilation and `git diff --check` pass.

No model, network, cluster, or paid call was made. The v4 gate remains unopened
until this implementation is committed and pushed, after which one exact rerun is
authorized.
