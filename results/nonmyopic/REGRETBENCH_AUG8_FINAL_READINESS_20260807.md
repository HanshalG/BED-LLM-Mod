# RegretBench Aug 8 Final Readiness

Date checked: 2026-08-07.

This is a zero-call, zero-write future-clock preflight of the exact Aug 8
execution chain from pushed commit `f2d922ca`. It changes no experimental
protocol, binding, threshold, model, seed, endpoint, or budget.

## Authenticated Account State

OpenRouter `/credits` returned:

- total credits: `$245.000000000`;
- total usage: `$220.113606154`;
- available balance: `$24.886393846`.

The user-reported `$30` top-up is still not posted and is not counted. The
authenticated balance nevertheless covers the exact `$4.80` Aug 8 chain cap.

## Exact Preflight Result

The preflight used an injected `2026-08-08 00:05 Europe/London` clock and the
registered `/opt/anaconda3/bin/python` interpreter. It returned:

- status: `ready_without_paid_calls`;
- next stage: `baseline_smoke`;
- model: `openai/gpt-5.6-luna`;
- baseline cap: `$0.20`;
- account-wide daily cap: `$5.00`;
- remaining after reserving the full baseline cap: `$4.80`;
- model calls made: `0`;
- files written: `0`.

The live catalog still reports Luna at `$0.10/M` prompt and `$0.60/M`
completion tokens, with structured outputs, reasoning, and the required input
modalities available. The `$0.008` request bound covers `30,848` prompt tokens
at the live price.

## Frozen Bindings

| Artifact | SHA-256 |
|---|---|
| `scripts/regretbench_aug8_execute.py` | `6e09bccdacc6da538b97eea86c90f77e6d054eaea382b12400ddb95187a949f8` |
| `results/nonmyopic/regretbench_aug8_chain/EXECUTION_BINDING.json` | `8c64bc2a538a72a7be71c47d2a495cb168a725c4ced62c5f46412ded150798fb` |
| Luna baseline daily executor | `db8d13229444abbf3f549967af4e70271103b6f5abd917fcce88573ec44b95c9` |
| DeepSeek support daily executor | `ad8c3f0ad907ea1042c08607976cca4bf900dc533347421806d7385616bf6eb7` |
| DeepSeek policy daily executor | `67d22c89385765bd0651a056c74d2960f70e13960f9bbf225dbc8e6ad0cb868c` |

All hashes match the active heartbeat. Every dated baseline, support, policy,
and daily-ledger output path is pristine.

## Verification

```text
24 passed in 0.51s
```

The tests cover the single-chain state machine and the support and policy daily
executors. The system Python lacks the repository's NumPy dependency; this is
not an execution blocker because the frozen command and heartbeat explicitly
use `/opt/anaconda3/bin/python`, which passed the preflight and tests.

## Next Action

On Aug 8, source `.env`, run the exact wrapper with `--preflight`, and execute
the same wrapper once only if it again returns `ready_without_paid_calls`. Do
not invoke component executors separately. The wrapper must stop descendants on
the first literal null or mechanics failure and reconcile account-wide spend
against the `$5.00` London-day cap.
