# Bongard OpenWorld Luna Development64 Execution

The active protocol is the prospective 64-task amendment. The executable file
retains its historical `development32` name for import and CLI compatibility,
but it now binds only:

`results/nonmyopic/bongard_openworld_luna_vlm_development64/PROTOCOL_MANIFEST_V14.json`

Manifest SHA-256:

`377596232d9fda34753bd99914292043ecc80a5584d55d95075e659c40e88011`

Run from the repository root after exporting `.env`. Each block is valid only
on its listed Europe/London date and only after the August 10 mechanics wrapper
and every earlier development block replay cleanly:

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

Each block contains 16 tasks: the original eight assigned to that block,
followed by eight byte-clean additions. It accepts exactly 528 first-stage
responses and 64--160 distinct terminal responses, with at most 688 accepted
responses, 702 HTTP attempts, and `$2.808` precharged exposure. The same-day
reasoning baseline contains 16 requests and at most `$0.128` exposure, so the
combined worst-case daily exposure is `$2.936`, below the hard account-wide
`$5` cap.

Block D opens endpoints only after all four endpoint-blind blocks independently
replay. Then bank the claim and exact confirmation handoff with:

```bash
/opt/anaconda3/bin/python scripts/bongard_openworld_luna_development_claim_finalize.py
```

Only the literal
`full_path_dependent_llm_native_development_signal` tier can authorize the
already frozen Confirmation96 protocol. A null or partial tier permanently
forbids confirmation execution.
