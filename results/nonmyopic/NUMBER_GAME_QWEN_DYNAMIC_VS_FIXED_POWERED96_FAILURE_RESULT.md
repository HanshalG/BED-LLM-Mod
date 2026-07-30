# Number Game Qwen Dynamic-vs-Fixed Powered-96 Failure Result

Date: 2026-07-30

## Status

**Failed closed before scientific scoring.** The run banked 13 complete trees,
then one of the 16 concurrent Gemini validation requests for tree index 13
returned `finish_reason="error"` on the original request and all four
transparent retries. The runner emitted `FAILURE.json` and stopped.

This is a transport failure, not a scientific null or positive result. No
aggregate endpoint was constructed, no partial-tree efficacy was inspected,
and none of the 13 complete trees may be reused in a successor experiment.

## Accounting

- `1,609` accepted responses before failure:
  - `1,372` Qwen 3.7 Plus;
  - `237` Gemini 2.5 Flash.
- `13` complete trees plus `114/115` accepted responses for the partial
  fourteenth tree.
- Every accepted response ended with `stop`.
- Zero reasoning tokens.
- Accepted-response ledger cost: `$2.03032588`.
- Provider-visible balance after failure: `$29.462566969`.

The failure happened in `_generate_supports` after the full planning tree and
15 of 16 validation responses had completed. The existing seeded adapter
retries a provider-error response with the identical seed and payload. This
means a deterministic seed-specific provider failure can exhaust all retries
even when the service is otherwise healthy.

## Consequence

The frozen powered-96 experiment is permanently `failed_closed`; it will not
be continued, repaired, or scored. A successor would require:

1. a separately frozen set of entirely new scientific seeds;
2. a deterministic, preregistered fallback-seed schedule used only after a
   zero-cost provider-error response;
3. a fresh validator-path serving smoke; and
4. the same policy, endpoint, model, support, and efficacy gates.

This transport-only amendment would not change or reclassify the failed run.

## Artifacts

- Public failure summary:
  `results/nonmyopic/number_game_qwen_dynamic_vs_fixed_powered96/number-game-qwen-dynamic-vs-fixed-powered96-20260730T013000Z/RESULT.json`
- Runner `FAILURE.json` SHA-256:
  `ff9547df0dff4f852f6d58480cd13990d255a9f317be43efce862df5c62157b3`
- Private partial raw responses SHA-256:
  `ee156a6b5c3c8309fc723375e4b42ef6a1eac3ea00f4b238ad2db9045d66fdd0`
