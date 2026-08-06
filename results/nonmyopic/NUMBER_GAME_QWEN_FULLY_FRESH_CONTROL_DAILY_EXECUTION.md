# Fully Fresh Qwen Control Daily Execution

Date initially frozen: 2026-08-06

Budget-utilization amendment frozen: 2026-08-06, before any control seed was
opened. This amendment authorizes only the two named reliability tails below;
it does not change the control calls, artifacts, verifier, or endpoint.

## Purpose

Execute the already-authorized 3,072-call history-blind control on the first
eligible later London day, reconcile the `$5/day` account-wide ledger, and run
the independent zero-call verifier. This wrapper does not change the frozen
source, control, or scientific gate implementations.

## Sequencing

The August 7 ledger opening usage is frozen at `$217.297890263`, the
authenticated cumulative usage after August 6 was closed to further calls.
This previous-day baseline is intentionally conservative: any account-wide
usage before the control counts against August 7 rather than being hidden in a
new opening snapshot.

The underlying daily-stage runner still enforces:

- a calendar date later than the August 6 source;
- the hash-bound source and structural authorization;
- `$4.25` projected allowance within a `$5.00` daily cap;
- at least `$5.00` live balance;
- an empty control output directory;
- exact registered control mechanics and seeds.

## Reconciliation

After the control returns, the executor immediately records the larger of:

- prior locally recorded spend plus measured control cost; or
- live posted usage minus the frozen daily opening usage.

Thus provider posting lag cannot undercount the run, while unrelated
account-wide usage cannot escape the daily cap. The ledger is updated before
verification, so a verifier failure cannot lose the paid spend record.

Only a complete, mechanics-passing control invokes the zero-call independent
verifier. Mechanics failure is banked without attempting scientific replay.
After a complete mechanics-passing control is independently verified and its
spend reconciled, the executor may authorize exactly these two unrelated tail
blocks:

- `openai/gpt-5.6-luna`, reliability128 interface, maximum `$0.10`;
- `deepseek/deepseek-v4-flash-0731`, reliability128 interface, maximum `$0.10`.

The authorizations are absent after an incomplete, mechanics-failed, or
unverified control, and absent if less than `$0.20` remains. Each reliability
runner consumes its exact pending authorization and still checks live
account-wide spend. The maximum authorized control-day total is therefore
`$4.25 + $0.20 = $4.45`, below the `$5.00` cap. No other paid tail is
authorized by this amendment.

## Command

Run only on or after 2026-08-07 Europe/London:

```bash
set -a
source .env
set +a
/Users/hanshalgoyal/.conda/envs/20_questions_env/bin/python \
  scripts/number_game_qwen_fully_fresh_control_daily_execute.py \
  --run-dir results/nonmyopic/number_game_qwen_fully_fresh_daily_stages/number-game-qwen-fully-fresh-daily-stages-20260806T000200Z \
  --run-id number-game-qwen-fully-fresh-daily-stages-20260806T000200Z \
  --daily-ledger results/nonmyopic/openrouter_daily_budget/2026-08-07.json
```

Expected paid cost is approximately `$3.21`; the hard block cap is `$4.25` and
the account-wide day cap is `$5.00`.
