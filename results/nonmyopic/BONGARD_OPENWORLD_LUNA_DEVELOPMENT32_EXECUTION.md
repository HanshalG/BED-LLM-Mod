# Bongard OpenWorld Luna Development32 Execution

## Fixed Daily Commands

Run from the repository root after loading `.env`. Each command is valid only
on its listed Europe/London date:

```bash
set -a
source .env
set +a

# 2026-08-11
/opt/anaconda3/bin/python scripts/bongard_openworld_luna_development32_daily_execute.py --block a --preflight
/opt/anaconda3/bin/python scripts/bongard_openworld_luna_development32_daily_execute.py --block a

# 2026-08-12
/opt/anaconda3/bin/python scripts/bongard_openworld_luna_development32_daily_execute.py --block b --preflight
/opt/anaconda3/bin/python scripts/bongard_openworld_luna_development32_daily_execute.py --block b

# 2026-08-13
/opt/anaconda3/bin/python scripts/bongard_openworld_luna_development32_daily_execute.py --block c --preflight
/opt/anaconda3/bin/python scripts/bongard_openworld_luna_development32_daily_execute.py --block c

# 2026-08-14
/opt/anaconda3/bin/python scripts/bongard_openworld_luna_development32_daily_execute.py --block d --preflight
/opt/anaconda3/bin/python scripts/bongard_openworld_luna_development32_daily_execute.py --block d
```

The driver uses the frozen protocol manifest at
`results/nonmyopic/bongard_openworld_luna_vlm_development32/PROTOCOL_MANIFEST.json`
and the fixed block directories and daily ledgers. Command-line overrides are
not exposed by the production CLI.

Each `--preflight` command is read-only. It verifies the current manifest,
Luna endpoint, live balance, target paths, the August 10 authorization, and
every prior block's result, ledger, and daily wrapper. Before a predecessor is
available it returns `waiting_for_aug10` or `waiting_for_block_<id>` rather
than creating files. Partial, tampered, or out-of-order predecessors fail.

The executable development interface is v4. Each block has exactly 264
first-stage requests: eight roots, 128 answer-conditioned branches, and 128
paired history-blind branches. It then generates 32--80 distinct final
histories, for 296--344 total requests. Interface-v2/v3 artifacts are invalid.
The current manifest SHA is recorded by the August 10 preflight result and
bound directly in the execution wrapper.

## Execution Guarantees

- Every block revalidates the passed August 10 wrapper and mechanics result.
- Before a fresh block writes its ledger or opens the adapter, it verifies a
  pristine target, the live Luna image/structured-output endpoint and pricing,
  and at least the full `$5.00` daily balance. It requires
  `ready_without_paid_calls` and freezes that exact live snapshot as the ledger
  opening boundary.
- Blocks B--D require every preceding block result, reconciled ledger, and
  `DAILY_EXECUTION.json` to replay and match their recorded hashes.
- Blocks A--C fail if a combined endpoint result exists. Their daily records
  assert that endpoint, confirmation, and sealed-test data remain unopened.
- Block D first replays all four endpoint-blind blocks. Only then may it run
  the combined endpoint analysis.
- Every replay verifies paired request seeds, prompt hashes, adjacent
  conditioned-then-blind dispatch in one 24-request batch, and that blind
  prompts contain only the initial four labels.
- Confirmation authorization co-requires the frozen dynamic-versus-history-blind
  history-change, Brier, bootstrap, log-loss, and ranking-fidelity gates.
- The combined result is recomputed independently in a temporary directory and
  must match canonically before any confirmation preregistration is authorized.
- Re-running a completed daily command revalidates banked artifacts and never
  repeats that block's model calls. A missing or changed completed artifact
  fails closed instead of being reconstructed silently.
- Every block has its own account-wide `$5.00` Europe/London ledger and a
  `$4.75` run cap. Unspent allowance does not roll over.

The complete Bongard regression suite passes `87` tests. Real dependency-aware
preflights for Blocks A--D on 2026-08-06 all returned `waiting_for_aug10`,
bound manifest `9c8c380c...73c6d0`, saw Luna at `$0.10/$0.60` per million
tokens and balance `$27.702109737`, and made zero calls or files. The waiting
status is the correct scientific predecessor gate until the banked August 10
result exists.

The driver can authorize only confirmation **preregistration**. It never
authorizes or executes confirmation tasks.

## Frozen Claim Report

After block D has produced and independently replayed the combined result, run
the zero-call claim classifier once:

```bash
python scripts/bongard_openworld_luna_claim_report.py \
  --combined-result results/nonmyopic/bongard_openworld_luna_vlm_development32/COMBINED_RESULT.json \
  --block-result results/nonmyopic/bongard_openworld_luna_vlm_development32/block-a-20260811/RESULT.json \
  --block-result results/nonmyopic/bongard_openworld_luna_vlm_development32/block-b-20260812/RESULT.json \
  --block-result results/nonmyopic/bongard_openworld_luna_vlm_development32/block-c-20260813/RESULT.json \
  --block-result results/nonmyopic/bongard_openworld_luna_vlm_development32/block-d-20260814/RESULT.json \
  --output results/nonmyopic/bongard_openworld_luna_vlm_development32/CLAIM_REPORT.json
```

The report can emit only the four pre-outcome tiers in
`BONGARD_OPENWORLD_LUNA_CLAIM_DECISION_PLAN.md`. Partial policy or mechanism
evidence cannot authorize confirmation or be presented as a full LLM-native
development signal.
