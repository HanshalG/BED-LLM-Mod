# Number Game Qwen History-Blind Serving Smoke Result

Date: 2026-07-30

## Status

**All frozen serving gates pass.** The matched 32-tree history-blind control is
authorized after its runner binds this public result hash. The smoke contains
no efficacy endpoint and none of its hypotheses are reused in the formal run.

## Exact-10 Gate

- model: `qwen/qwen3.7-plus`, reasoning disabled;
- prompt: initial Number Game prompt with no observations;
- seeds: `8900000..8900009`;
- accepted requests / HTTP attempts: `10 / 10`;
- retries / provider-error retries: `0 / 0`;
- reasoning tokens / forced exits: `0 / 0`;
- parser: `10/10` strict JSON draws;
- valid unique rules per draw: `21..23`;
- pooled unique rules: `29..32`;
- second-draw novel contributions: `6..10`;
- cost: `$0.01028416`, below the frozen `$0.12` cap.

All ten draws exceed the 16-rule floor, every pooled support exceeds the
24-rule floor, and every second draw adds at least two extensions.

## Decision

Hash-bind the smoke into the already frozen formal runner. Open the formal
`9000000+` seeds once only. Do not alter its cohort, prompts, endpoints,
scientific gates, request count, concurrency, or cost cap based on this smoke.

## Provenance

- public `RESULT.json` SHA256:
  `0e4ba2e5bedd000d5b22c46e16a91404ec8913ba620ab54db286309fef9b8ef4`
- private raw-response SHA256:
  `33790beba5a2fba5861dcee7bda4483def2c3fb45179a18feba3eb1445f57011`
- private raw responses remain uncommitted.
