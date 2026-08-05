# Number Game Qwen Fully Fresh Daily-Stages Implementation

Date completed: 2026-08-05

## Status

The budget-compatible staged runner is ready. No formal source or control seed
has been opened and this implementation used zero model calls.

## Implementation

`scripts/number_game_qwen_fully_fresh_daily_stages.py` provides two fail-closed
commands over the unchanged fully fresh source and control engines:

```bash
python scripts/number_game_qwen_fully_fresh_daily_stages.py source ...
python scripts/number_game_qwen_fully_fresh_daily_stages.py control ...
```

The source command requires a current Europe/London ledger with the full
`$5.00` allowance, tightens the source engine and mechanics cap to `$5.00`,
and binds the original source artifacts into `CONTROL_AUTHORIZATION.json`.
Authorization uses only source mechanics and the changed-root floor. A
structural failure writes the final mechanics/opportunity result immediately.

The control command requires a later calendar date and a fresh `$4.25`
allowance. It verifies the authorization-file and source-artifact hashes,
recomputes structural authorization, and runs regardless of source scientific
gates. It uses the original composite finalizer under a stricter `$9.25` cap
and preserves the independent replay entrypoint.

## Verification

The staged/original orchestration suite passes 20 tests. Adversarial coverage
includes authorization invariance to source efficacy pass versus null, budget
and same-day refusal before calls, source and manifest tamper refusal,
structural failure with no control directory, and later-day control execution
despite source-science null.

The full source/control family passes 44 tests in 114.93 seconds, including the
resilient source engine, all history-blind control variants, first-link
confirmation, composite orchestration, and replay machinery.

## Next Action

2026-08-05 has only `$4.504258308` remaining, below the exact source-stage
reservation. The source command must be executed on the next day with a fresh
`$5.00` ledger. No paid run is authorized before that calendar reset.
