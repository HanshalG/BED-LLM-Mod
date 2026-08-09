# OpenRouter August 9 Daily Budget And Frontier Amendment

Date: 2026-08-09 (Europe/London)

Status: zero-call budget and model-routing amendment. This note refines
`OPENROUTER_MODEL_PARETO_AND_DAILY_ROUTE_20260808.md`; it changes no frozen
model, prompt, reasoning effort, task, seed, endpoint, threshold, request count,
or paid-stage authorization.

## Live Account Boundary

An authenticated read after sourcing `.env` reports:

- total credits: `$245.000000000`;
- cumulative usage: `$220.121013787`;
- available balance: `$24.878986213`;
- current key daily usage: `$0.000000000`.

The newly reported `$30` top-up is not yet visible in OpenRouter's `/credits`
response. It therefore supplies no current authorization. If it posts without
intervening use, the available balance will be `$54.878986213`, enough for ten
complete `$5` London-day ceilings plus `$4.878986213`.

The operating rule remains a hard account-wide `$5.00` cap per Europe/London
calendar day. Unrelated account use counts. A new ledger snapshots cumulative
usage at day opening; every paid stage is authorized immediately before dispatch;
and reconciliation uses the larger of posted usage since opening and locally
measured accepted-request cost. Unused allowance does not roll over.

## What “Use Five Dollars Properly” Means

The cap is capacity for the strongest dependency-valid evidence, not a quota that
must be burned. The frozen Bongard chain already allocates nearly the full cap on
the expensive days:

| London date | Frozen work | Run-level authorization | Attempt-level precharged exposure |
|---|---|---:|---:|
| Aug 10 | serving plus Mechanics4 | `$0.25 + $1.75 = $2.00` | at most `$0.720` for the bounded mechanics attempts |
| Aug 11--14 | one Development16 block plus its paired Luna-medium naive baseline | `$4.75 + $0.20 = $4.95` per day | at most `$2.876 + $0.128 = $3.004` per day |
| Aug 15--18 | one Confirmation24 block, only if development authorizes confirmation | `$4.75` per day | at most `$4.312` per day |

Run-level authorization is deliberately larger than the sum of bounded
per-attempt charges. It is the ledger ceiling; attempt precharge is the maximum
simultaneous HTTP exposure under the frozen retry schedule. Actual spend is
reconciled after each stage and may be lower.

Across the complete mechanics, four development-plus-baseline days, and four
confirmation days, the attempt-level exposure is at most `$29.984`. The run-level
daily allocation is `$40.80`, but it cannot force model output or manufacture
scientifically invalid work. On Aug 10, or after any preregistered gate closes a
branch, the unused balance stays unused unless a separately frozen same-claim
dependency is ready before any endpoint is seen.

## Exact Live Model Frontier

OpenRouter's authenticated catalog currently reports:

| Exact route | Input / output per 1M tokens | Context | Modalities |
|---|---:|---:|---|
| `deepseek/deepseek-v4-flash-0731` | `$0.09 / $0.18` | `1,048,576` | text only |
| `openai/gpt-5.6-luna` | `$0.10 / $0.60` | `1,050,000` | text, image, file |

The DeepSeek route has 24 live provider endpoints. The catalog-selected top
provider exposes a `384,000`-token completion ceiling. Luna's selected OpenAI
endpoint exposes a `128,000`-token completion ceiling and the image capability
required by Bongard OpenWorld. Concurrency changes wall time, not the token bill;
the frozen Bongard protocol therefore keeps its validated concurrency rather than
raising it merely because account credit increased.

Artificial Analysis provides useful effort curves but not a clean, indexed page
for the exact July 31 `0731` revision. Its current generic V4 Flash pages report
scores around `29` nonreasoning, `37` high, and `40` max. Those values must not be
presented as an exact `0731` measurement. Luna's current indexed curve is `27`
nonreasoning, `39` medium, `46` high, and `49` xhigh. Generic benchmarks therefore
support escalation choices, but direct task gates remain decisive.

Sources:

- OpenRouter exact DeepSeek route:
  https://openrouter.ai/deepseek/deepseek-v4-flash-0731
- Artificial Analysis generic DeepSeek V4 Flash max:
  https://artificialanalysis.ai/models/deepseek-v4-flash
- Artificial Analysis Luna nonreasoning:
  https://artificialanalysis.ai/models/gpt-5-6-luna-non-reasoning
- Artificial Analysis Luna medium:
  https://artificialanalysis.ai/models/gpt-5-6-luna-medium
- Artificial Analysis Luna high:
  https://artificialanalysis.ai/models/gpt-5-6-luna-high
- Artificial Analysis Luna xhigh:
  https://artificialanalysis.ai/models/gpt-5-6-luna-xhigh

## Frozen Routing Decision

1. Use `deepseek/deepseek-v4-flash-0731` nonreasoning for high-volume text-only
   support, likelihood, and planning after exact schema and semantic gates.
2. Use DeepSeek high, then max, only for small hard text gates with fixed output
   caps and an observed cost projection. Its low token price does not repair a
   failed semantic-support gate.
3. Use `openai/gpt-5.6-luna` nonreasoning for bulk visual beliefs. DeepSeek never
   receives Bongard images.
4. Use Luna medium only for the separately labelled thinking/naive visual
   baseline.
5. Escalate difficult visual tasks to Luna high before xhigh. Do not silently
   enable reasoning or swap models after observing a scientific endpoint.

This routing matches direct project evidence: DeepSeek is the economical text
worker but has previously produced schema-valid, semantically inadequate support;
Luna medium passed the exact multimodal naive smoke `10/10`. The unopened Bongard
protocol remains Luna nonreasoning for visual belief dynamics plus its frozen
Luna-medium baseline.

The focused executable budget suite passes `55/55`, covering the generic daily
ledger, Aug 10 wrapper, development block, paired naive baseline, and confirmation
block.

This amendment made zero paid model calls and spent `$0.00`.
