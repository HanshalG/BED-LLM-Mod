# Bongard OpenWorld Luna Development32 Execution

## Fixed Daily Commands

Run from the repository root after loading `.env`. Each command is valid only
on its listed Europe/London date:

```bash
set -a; source .env; set +a

# 2026-08-11
python scripts/bongard_openworld_luna_development32_daily_execute.py --block a

# 2026-08-12
python scripts/bongard_openworld_luna_development32_daily_execute.py --block b

# 2026-08-13
python scripts/bongard_openworld_luna_development32_daily_execute.py --block c

# 2026-08-14
python scripts/bongard_openworld_luna_development32_daily_execute.py --block d
```

The driver uses the frozen protocol manifest at
`results/nonmyopic/bongard_openworld_luna_vlm_development32/PROTOCOL_MANIFEST.json`
and the fixed block directories and daily ledgers. Command-line overrides are
not exposed by the production CLI.

The executable development interface is v3 and the corrected manifest SHA is
`d5e8412f6e2f485a357ba255692f1c6d60a99b4900e588b05ad39b9f276b5b9c`.
Interface-v2 artifacts are invalid.

## Execution Guarantees

- Every block revalidates the passed August 10 wrapper and mechanics result.
- Blocks B--D require every preceding block result, reconciled ledger, and
  `DAILY_EXECUTION.json` to replay and match their recorded hashes.
- Blocks A--C fail if a combined endpoint result exists. Their daily records
  assert that endpoint, confirmation, and sealed-test data remain unopened.
- Block D first replays all four endpoint-blind blocks. Only then may it run
  the combined endpoint analysis.
- The combined result is recomputed independently in a temporary directory and
  must match canonically before any confirmation preregistration is authorized.
- Re-running a completed daily command revalidates banked artifacts and never
  repeats that block's model calls. A missing or changed completed artifact
  fails closed instead of being reconstructed silently.
- Every block has its own account-wide `$5.00` Europe/London ledger and a
  `$4.75` run cap. Unspent allowance does not roll over.

The driver can authorize only confirmation **preregistration**. It never
authorizes or executes confirmation tasks.
