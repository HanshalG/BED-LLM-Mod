# Number Game Validator Fallback Serving Smoke Failure

Date: 2026-07-30

## Status

**Failed closed in the smoke evaluator.** All ten non-scientific Gemini
requests returned accepted `stop` responses, but the gate expected the pooled
item-isolated parser's `codec_mode` field. The validator path uses the base
strict parser, whose successful return is already strict and whose diagnostic
contains `raw_count`, `valid_unique_count`, and `rejected` instead.

The resulting `KeyError` occurred before `RESULT.json` serialization. The
runner wrote `FAILURE.json`; no scientific seed was opened and the resilient96
successor was not authorized.

## Accounting

- 10 accepted Gemini 2.5 Flash responses;
- all 10 finish reasons `stop`;
- zero reasoning tokens;
- accepted-response cost `$0.022189`;
- provider-visible balance `$29.454175469`.

The responses are intentionally not reconstructed or reused. A corrected
smoke must use fresh non-scientific seeds and a synthetic end-to-end parser
contract test before any paid call.

## Artifacts

- Public failure summary:
  `results/nonmyopic/number_game_validator_fallback_serving_smoke/number-game-validator-fallback-serving-smoke-20260730T020000Z/RESULT.json`
- Runner `FAILURE.json` SHA-256:
  `df347b37c27f1bae9d409c5a73ecf77b30abe3a421076a4c6afd50eb8a07f134`
