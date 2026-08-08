# RegretBench Aug 8 Current-Commit Readiness

Date checked: 2026-08-07.

This is a zero-call, zero-write future-clock audit of the exact Aug 8 execution
chain on pushed commit `6e908b78ac6048697af647195b09452e29ba922f`. It
supersedes only the stale commit reference in the earlier readiness note. It
does not change any protocol, binding, threshold, model, seed, endpoint, path,
or budget.

## Authenticated Account State

OpenRouter `/credits` returned the same byte-for-byte payload before and after
the preflight:

- total credits: `$245.000000000`;
- total usage: `$220.113606154`;
- available balance: `$24.886393846`.

The user-reported `$30` top-up is still not posted and is not counted. The
authenticated balance covers the exact `$4.80` Aug 8 chain cap. The hard
account-wide Europe/London calendar-day ceiling remains `$5.00`, with no
borrowing or rollover.

## Exact Preflight Result

The preflight used an injected `2026-08-08 09:00 Europe/London` clock and
`/opt/anaconda3/bin/python`. It returned:

- status: `ready_without_paid_calls`;
- next stage: `baseline_smoke`;
- model: `openai/gpt-5.6-luna`;
- baseline cap: `$0.20`;
- account-wide daily cap: `$5.00`;
- remaining after reserving the full baseline cap: `$4.80`;
- model calls made: `0`;
- files written: `0`.

The live catalog reports Luna at `$0.10/M` prompt and `$0.60/M` completion
tokens, with file, image, and text inputs plus structured outputs and reasoning.
The maximum completion length is `128,000`. The frozen `$0.008` request bound
covers `30,848` prompt tokens at the live price.

All six dated paid-path families remain absent and pristine: Luna baseline
smoke; DeepSeek support smoke and development; DeepSeek policy enriched smoke,
optional naive smoke, and development.

## Frozen Chain Bindings

| Artifact | SHA-256 |
|---|---|
| `scripts/regretbench_aug8_execute.py` | `6e09bccdacc6da538b97eea86c90f77e6d054eaea382b12400ddb95187a949f8` |
| `results/nonmyopic/regretbench_aug8_chain/EXECUTION_BINDING.json` | `8c64bc2a538a72a7be71c47d2a495cb168a725c4ced62c5f46412ded150798fb` |

These hashes match the active heartbeat. The later zero-call horizon-value
diagnostic and conditional Aug 9 factorized contingency do not modify this
chain and cannot authorize or rescue an Aug 8 result.

## Verification

The exact orchestrator, support daily executor, and policy daily executor tests
pass:

```text
24 passed in 0.46s
```

The current full RegretBench suite also passed `235/235` after the latest
zero-call diagnostic was added. No paid request was made during either audit.

## Next Action

On Aug 8, source `.env`, run exactly:

```bash
/opt/anaconda3/bin/python scripts/regretbench_aug8_execute.py --preflight
```

Only if it again returns `ready_without_paid_calls`, run the same command once
without `--preflight`. Do not invoke component executors separately. The
wrapper must stop descendants on the first literal null or mechanics failure
and reconcile total account-wide spend against the `$5.00` London-day cap.
