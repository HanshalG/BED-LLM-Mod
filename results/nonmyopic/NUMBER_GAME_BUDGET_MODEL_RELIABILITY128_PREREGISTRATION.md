# Number Game Budget-Model Reliability128 Preregistration

Date frozen: 2026-08-06

## Purpose

Determine whether GPT-5.6 Luna and DeepSeek V4 Flash 0731 are mechanically
reliable enough to enter a paired Number Game efficacy experiment. This is a
model/interface gate, not a scientific efficacy endpoint.

The gate targets the observed model-specific failures:

- Luna passed exact-10, but its unchanged scale attempt recorded 15 forced
  4,200-token exits among 1,200 accepted calls and then failed strict JSON;
- DeepSeek 0731 completed exact-10 transport and top-level parsing, but one
  conditioned response yielded zero valid executable hypotheses.

No target concepts, Brier scores, selected roots, or policy comparisons are
used for authorization.

## Frozen Models And Interface

- `openai/gpt-5.6-luna`, eight request seeds `1080801..1080808`;
- `deepseek/deepseek-v4-flash-0731`, eight request seeds
  `1080821..1080828`;
- nonreasoning, temperature `0.7`, strict existing 24-item JSON schema;
- existing executable-rule parser and consistency filter, unchanged;
- maximum 4,200 output tokens;
- eight matched seed groups, each with one initial, five one-observation, and
  ten two-observation calls;
- concurrency `8` per seed group and `64` in aggregate;
- one same-prompt, same-temperature retry only after strict parse failure;
- no retry for low valid-support count or any semantic failure.

The retry is an operational reliability measurement, not silent repair. Raw
parse failures, retry calls, forced exits, and final outcomes remain separate
public metrics. More than three initial parse failures closes the model without
format retries.

## Frozen Cases

Exactly 128 calls per model before format retries:

- 8 no-observation calls;
- 40 distinct one-observation histories;
- 80 distinct two-observation histories.

Conditioned histories are selected by SHA-256 rank with seed `1080800` from
the unique branch histories in the fully fresh Qwen source artifact:

`results/nonmyopic/number_game_qwen_fully_fresh_daily_stages/number-game-qwen-fully-fresh-daily-stages-20260806T000200Z/source/TREES.json`

Bound source SHA-256:
`f812e4a356f5129a6f2f22d5f0f995b76b4ac624a0f312154064ff8c51f2c7b0`.

Repeated initial prompts intentionally measure stochastic interface
reliability. Histories are used only to construct prompts; stored generated
supports and all efficacy endpoints are ignored.

## Passing Gates

All must pass independently for a model:

- exactly 128 initial accepted requests and exactly 128 final case records;
- at most three initial strict-parse failures;
- every initial parse failure succeeds after its single same-prompt retry;
- accepted requests equal `128 + format_retry_requests`;
- HTTP attempts equal accepted requests plus transport retries;
- at most four transport retries and zero provider-error retries;
- zero reasoning tokens;
- at most three forced exits across initial and retry calls;
- every parsed response has exactly 24 schema items;
- every no-observation response has at least 16 valid unique hypotheses;
- every conditioned response has at least four valid unique hypotheses;
- mean conditioned valid support is at least eight;
- measured model cost is at most `$0.10`.

A response that parses but has insufficient valid hypotheses is final and
cannot be retried. A model may advance only if every gate passes.

## Budget And Sequencing

Each model has a `$0.10` cap; the combined screen is capped at `$0.20`.
Execution requires a current Europe/London daily ledger and a completed,
mechanics-passing fully fresh Qwen history-blind control with exactly 3,072
accepted control calls. The CLI verifies that result before constructing an
adapter. The screen is scheduled for a later daily block and makes no calls on
2026-08-06.

If both models pass, both may enter separately frozen paired efficacy gates;
the reliability screen itself does not select by benchmark score or price. If
only one passes, only that model advances. If neither passes, the unchanged
interfaces close and no scale run is authorized.
