# OpenRouter Budget And Budget-Model Plan

Date checked: 2026-08-06 (Europe/London)

## Enforceable Balance

The authenticated credits endpoint still reports `$245.00` credited and
`$217.297890263` used, leaving `$27.702109737`. The newly reported `$30`
credit is not visible yet and is excluded from every authorization
calculation until it posts.

- Posted balance: five complete `$5` daily authorizations plus `$2.702109737`.
- Balance after the reported credit posts: `$57.702109737`, or eleven complete
  `$5` daily authorizations plus `$2.702109737`.
- Daily cap: `$5.00` account-wide in Europe/London, including unrelated use.
- No rollover and no reserve. The cap is a safety ceiling, not an instruction
  to buy uninformative calls.
- Every paid driver reads live cumulative usage immediately before a block,
  reconciles actual cost afterward, and fails closed on provider lag or
  unrelated account use.

Today is closed at `$4.26043712`; no additional August 6 paid block is
authorized.

The zero-call August 7 preflight passes against the authenticated account and
frozen artifacts. Its full expected sequence is `$4.96`, leaving `$0.04` under
the hard daily cap. Stress is guaranteed when the measured control cost is at
most `$3.25`; if the control is slightly higher, a science-blind ledger-only
reconsideration after both reliability gates can authorize stress when total
recorded spend remains at most `$3.45`.

## Frozen Daily Sequence

| London date | Scientific block | Maximum / expected spend |
|---|---|---:|
| Aug 7 | Qwen history-blind control, Luna+0731 reliability128, conditional selected-model stress3584 | `$5.00` / `$4.96` |
| Aug 8 | Number Game diversity-bonus confirmation block A | `$5.00` cap |
| Aug 9 | Number Game diversity-bonus confirmation block B, only after A mechanics pass | `$5.00` cap |
| Aug 10 | Bongard exact-10 serving then four-task mechanics, only if serving passes | `$2.00` component-cap sum |
| Aug 11 | Bongard development block A, only after Aug 10 mechanics pass | `$4.75` cap |
| Aug 12 | Bongard development block B, with prior-block replay | `$4.75` cap |
| Aug 13 | Bongard development block C, with prior-block replay | `$4.75` cap |
| Aug 14 | Bongard development block D, then endpoint opening | `$4.75` cap |

The maximum listed component caps sum to `$35.96`. Execution remains gated
one day at a time; a null or incomplete predecessor closes its descendants.
The unused `$3` on Aug 10 is not automatically filled with an unregistered
experiment.

## Model Decision

### GPT-5.6 Luna

- Exact endpoint: `openai/gpt-5.6-luna`.
- Live OpenRouter price: `$0.10/M` input and `$0.60/M` output.
- Input: text, images, and files; strict structured output is advertised.
- Context: 1.05M; maximum completion: 128K.
- Artificial Analysis: 51 at max reasoning, with 130M output tokens across its
  Intelligence Index evaluation. This is a strength signal, not a projection
  of our non-reasoning semantic-support cost.
- Direct project evidence: passed the exact ten-request Number Game semantic
  support smoke for `$0.0035776`, but a later scale attempt had 15 forced
  length exits in 1,200 responses and a fatal strict-JSON failure.

Decision: primary multimodal model and current semantic-generator favorite,
subject to the frozen reliability128 gate. Use non-reasoning for the BED
belief process; reasoning remains a separately labelled naive-thinking
baseline.

### DeepSeek V4 Flash 0731

- Exact endpoint: `deepseek/deepseek-v4-flash-0731`.
- Live OpenRouter price: `$0.09/M` input and `$0.18/M` output.
- Text-only; structured output and reasoning controls are advertised.
- Context: 1,048,576; maximum completion: 65,536.
- Architecture: 284B total / 13B active. It is a July 31 re-post-training of
  V4 Flash. Artificial Analysis now reports an exact-checkpoint score of 50 at
  max reasoning, 103.3 output tokens/s, and 210M output tokens across the
  Intelligence Index evaluation.
- Direct project evidence: strict parsing and transport were clean in exact
  ten, but one conditioned response yielded zero valid executable hypotheses,
  so the frozen support gate failed. Cost was `$0.003258592`; high verbosity
  erased most of its nominal price advantage over Luna.

Decision: cheapest serious text challenger and thinking-baseline candidate,
not a Bongard candidate because it cannot consume images. It must pass the
same reliability gate before it owns a large semantic-generation block.

## Next Model Gate

After the mandatory Aug 7 Qwen control, run 128 matched reliability cases for
Luna and 0731 (`$0.10` cap each). Select by strict parse completion, forced
exits, conditioned support validity, and support diversity, not generic
benchmark rank. Only the selected model may receive the conditional 3,584-case
stress block (`$1.55` cap), and only if reconciled account-wide spend leaves
its full cap. The authorization decision cannot inspect which model wins.

Sources: [OpenRouter Luna](https://openrouter.ai/openai/gpt-5.6-luna),
[OpenRouter DeepSeek 0731](https://openrouter.ai/deepseek/deepseek-v4-flash-0731),
[Artificial Analysis Luna](https://artificialanalysis.ai/models/gpt-5-6-luna-high),
and the [DeepSeek V4 model card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash).
