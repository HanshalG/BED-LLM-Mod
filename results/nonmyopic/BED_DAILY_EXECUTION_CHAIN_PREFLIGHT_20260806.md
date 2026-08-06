# BED Daily Execution Chain Preflight

Date: 2026-08-06 Europe/London

## Decision

The frozen August 7--14 execution chain is operationally ready at its current
dependency boundaries. All checks made zero model calls and wrote zero files.

| Date | Block | Read-only status |
|---|---|---|
| Aug 7 | Number Game control, budget-model reliability and stress | `ready_without_paid_calls` |
| Aug 8 | Number Game diversity confirmation A | `waiting_for_aug7_control` |
| Aug 9 | Number Game diversity confirmation B | `waiting_for_block_a` |
| Aug 10 | Bongard exact10 and mechanics | `ready_without_paid_calls` |
| Aug 11--14 | Bongard development A--D | `waiting_for_aug10` |

Waiting statuses are expected dependency gates, not launch failures. The live
authenticated OpenRouter balance remained `$27.702109737`; the reported `$30`
top-up was still absent.

## Audit Fix

The Bongard development wrapper previously exposed only an internal volatile
runtime check. It had no production `--preflight` command that jointly checked
the protocol manifest, August 10 authorization, prior daily blocks, live model
catalog, balance, and pristine target paths.

The wrapper now exposes:

```bash
/opt/anaconda3/bin/python \
  scripts/bongard_openworld_luna_development32_daily_execute.py \
  --block <a|b|c|d> --preflight
```

The preflight is dependency-aware and read-only. It returns a waiting status
for an absent legitimate predecessor and rejects partial, tampered, or
out-of-order artifacts. All four real invocations bind protocol manifest
`9c8c380cc6c5fe248cc06401bd4a7b4f160620f4eec449e246e05a404473c6d0`.

Number Game commands use the project conda environment. Bongard commands use
`/opt/anaconda3/bin/python`, which contains Pillow; the Number Game environment
does not and therefore cannot import the Bongard image pipeline.

## Verification

- dedicated development daily-wrapper tests: `17/17`;
- complete Bongard regression suite: `87/87`;
- model calls: `0`;
- files written by every preflight: `0`;
- scientific endpoints opened: none.
