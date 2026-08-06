# Number Game Aug 7 Budget-Model Screen Result

Date: 2026-08-07 (Europe/London)

## Result

The fully fresh Qwen control stopped at its frozen mechanics boundary after
exactly 3,072 accepted requests and `$2.69017344`. One of 1,536 pooled branches
missed the frozen draw/support floors. The policy endpoint was not opened, and
the failed control cannot authorize the Aug 8 diversity block.

The independently frozen model-only tail then evaluated both budget models on
128 reliability cases each:

| Model | Status | Requests | Cost | Parse failures | Forced exits | Conditioned valid min / mean | Conditioned draws below 4 |
|---|---:|---:|---:|---:|---:|---:|---:|
| GPT-5.6 Luna | gated null | 128 | `$0.05738300` | 4 | 4 | `0 / 15.925` | 3 / 120 |
| DeepSeek V4 Flash 0731 | gated null | 128 | `$0.03447770` | 0 | 0 | `0 / 12.108` | 20 / 120 |

Luna failed strict-output and support-floor gates. DeepSeek was cheaper and
transport/schema-clean, but many conditioned generations collapsed to
extension-equivalent or observation-inconsistent rules. Neither model was
eligible under the frozen ordering. The stress3584 selector therefore returned
`no_eligible_model` with zero requests and zero cost.

Total recorded Aug 7 spend is `$2.782034141`; `$2.217965859` remains under the
account-wide `$5` daily ledger. Model selection used no policy efficacy value.

## Interpretation

Under the unchanged Number Game support-generation interface, neither Luna nor
DeepSeek V4 Flash 0731 replaces Qwen. DeepSeek is the better development base:
it is cheaper, returned exact JSON on all 128 cases, and used no forced exits.
Its blocker is semantic support diversity and constraint satisfaction rather
than transport. Any prompt-diversity revision must be treated as a new
development interface and tested on fresh cases; these nulls remain immutable.

## Artifact Bindings

- daily execution wrapper:
  `7eeef7a9473e2898c66a90fec9dd389441a9afd302dbce9bf45d3596347a5f7e`;
- failure-tail wrapper:
  `1466fc6d8249b51564112b7d2200bf22987a6f562c197f1fda07b0ed15909a8b`;
- Luna result:
  `2949ab2faac0a2e4597cec95f2d70626a724ed753a8bb12c8f189aa149026915`;
- DeepSeek result:
  `c5843bbedc1780200717705281dd2cdb6a6f927492acf5cf9b2b0b0937403a1b`;
- no-model stress result:
  `62e26f11a1747f11411d2b366d07051e0257e1a9fdbd7e40b960af317579c5e3`;
- reconciled daily ledger:
  `15022d299a9ba09c7e57ce7e1097347cdf47e4650866beffd25de6ebb6f66ea7`.

Private raw responses are retained for exact replay and must not be used to
relax the frozen gates or score the unopened policy endpoint.
