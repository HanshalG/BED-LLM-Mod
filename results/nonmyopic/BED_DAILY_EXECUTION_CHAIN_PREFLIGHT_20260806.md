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
`8659fb5fc6a02ddc59eb7147b6663d1fef3f96880e29f5e6de9bc0386f8e24aa`.

A later transport audit found that the concurrent Luna adapters still relied
on post-response run-budget checks. August 10--14 interface-v2 wrappers now
bind precharge amendment `75acd7ae...bbbff4`, reserve `$0.004` before every
attempt, and verify 20,800 prompt-token coverage after the maximum 3,200-token
output at live prices. The rebound protocol manifest changes implementation
hashes only; tasks, seeds, prompts, batches, endpoints, and gates are unchanged.

Number Game commands use the project conda environment. Bongard commands use
`/opt/anaconda3/bin/python`, which contains Pillow; the Number Game environment
does not and therefore cannot import the Bongard image pipeline.

## Verification

- dedicated development daily-wrapper tests: `17/17`;
- complete Bongard regression suite: `92/92`;
- model calls: `0`;
- files written by every preflight: `0`;
- scientific endpoints opened: none.
