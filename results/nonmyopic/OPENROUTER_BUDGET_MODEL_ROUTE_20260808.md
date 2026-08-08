# OpenRouter $5/Day Budget And Model Route

Date: 2026-08-08 (Europe/London)

## Live Account Boundary

The authenticated `/api/v1/credits` response at this audit reports:

- total credits: `$245.000000000`;
- cumulative usage: `$220.121013787`;
- available balance: `$24.878986213`.

The newly reported `$30` top-up is not present in the authenticated credit total.
It is not available to any ledger until the provider posts it. If it posts without
intervening account usage, the balance becomes `$54.878986213`, enough for ten full
`$5` days plus `$4.878986213`.

The binding operating rule is a hard account-wide `$5.00` cap per London calendar
day. Every paid day starts from authenticated cumulative usage. A stage reserves its
full worst-case HTTP-attempt exposure before dispatch and is re-authorized immediately
before calls. Spend is the maximum of posted account usage since opening and locally
measured accepted-request cost. Unrelated account use counts. There is no borrowing or
rollover.

The target is useful spend close to `$5`, not nominal spend. The priority order is:

1. the next dependency-valid headline experiment;
2. its preregistered paired baseline or matched-compute control;
3. a direct task-interface model gate needed by the next experiment;
4. transport headroom.

A scientific null or failed gate closes descendants even when allowance remains. A
new, post-outcome endpoint is not invented solely to consume the remainder.

## Live Routes And Prices

The authenticated OpenRouter model catalog contains these dated routes:

| Route | Input / output per 1M | Context | Modalities | Frozen role |
|---|---:|---:|---|---|
| `deepseek/deepseek-v4-flash-0731` | `$0.09 / $0.18` | 1,048,576 | text | high-volume text-only nonreasoning planning/support |
| `openai/gpt-5.6-luna` | `$0.10 / $0.60` | 1,050,000 | text, image | visual semantic belief; labelled thinking baseline |

Use the dated DeepSeek ID rather than `~deepseek/deepseek-v4-flash-latest`, so a
provider checkpoint cannot change silently. DeepSeek must never receive Bongard
images. Existing frozen Bongard runs remain Luna runs; changing the model would change
the estimand and invalidate their gates.

At current OpenRouter prices, DeepSeek output is `3.33x` cheaper than Luna output.
For a representative request with 4,000 uncached input and 2,000 generated tokens,
the catalog-price estimates are `$0.00072` for DeepSeek and `$0.00160` for Luna. A
`$5` ceiling therefore buys about 6,944 versus 3,125 such requests before retry
headroom. Actual reasoning verbosity and provider-reported costs remain authoritative.

## Intelligence-Per-Dollar Screen

Artificial Analysis v4.1.1 reports DeepSeek V4 Flash 0731 at max reasoning with an
Intelligence Index of `52`, price `$0.14/$0.28` per million on its measured endpoint,
and approximately `210M` output tokens over the full evaluation. Its July 31 analysis
places the earlier v4.1 score at `50`, one point behind Luna max, with about 60% lower
cost per task on DeepSeek's first-party endpoint.

The same evaluator reports GPT-5.6 Luna high at `47`, 176 output tokens/s, and `37M`
output tokens over the evaluation. Its GPT-5.6 launch analysis reports Luna max at
`51` and about `$0.21` per Intelligence Index task before the subsequent price data
reflected by the current model page.

These broad reasoning results make both models Pareto candidates, but they do not
measure our deployed interface. DeepSeek's max-reasoning token volume warns that a
cheap token price is not automatically a cheap many-rollout planner. Conversely,
Luna's multimodal support and concision do not establish strict JSON reliability.

Sources:

- https://artificialanalysis.ai/models/deepseek-v4-flash
- https://artificialanalysis.ai/articles/deepseek-v4-flash-0731-scores-50-on-the-artificial-analysis-intelligence-index-10-points-above-previous-deepseek-v4-flash
- https://artificialanalysis.ai/models/gpt-5-6-luna-high/
- https://artificialanalysis.ai/articles/gpt-5-6-has-landed/
- https://api-docs.deepseek.com/quick_start/pricing/
- https://openai.com/index/gpt-5-6/

## Direct Project Evidence

The project evidence controls routing:

| Evidence | DeepSeek 0731 | GPT-5.6 Luna |
|---|---|---|
| Number Game exact planning scale | 1,568 accepted requests, `$0.3496433702`, zero reasoning tokens | 1,200 accepted requests, `$0.4889055`, then malformed JSON; no endpoint score |
| Number Game non-myopic signal | depth-three Brier `4.0619%` below own myopic, but missed frozen efficacy/mechanics gates | unavailable because strict run failed closed |
| Conditioned-support reliability128 | 128/128 structurally clean, but semantic support gate failed | 4 malformed/forced outputs and support gate failed |
| Bongard visual input | unsupported | supported; frozen route |

Decision:

- DeepSeek 0731 is the default budget model for new text-only, high-volume,
  nonreasoning support generation and planning.
- Luna is the default for image-conditioned semantic belief and the separately
  labelled thinking baseline.
- Reasoning is not silently enabled for planner or environment roles. Any reasoning
  comparison is a naive baseline or a separately preregistered model gate.
- Generic benchmark rank never overrides a failed direct mechanics or semantic-support
  gate.

No paid model call was made for this audit.
