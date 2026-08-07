# Bongard-OpenWorld Model Escalation Audit

Date: 2026-08-07

## Decision

Keep `openai/gpt-5.6-luna` as the frozen August 10 model. Treat
`qwen/qwen3.7-plus` as the preregistered next model to investigate only after a
banked Luna serving or mechanics failure, not as an in-place substitute.

## Live Catalog

The authenticated OpenRouter catalog reports:

| Model | Input | Structured output | Context | Input / output price per M tokens |
| --- | --- | --- | ---: | ---: |
| GPT-5.6 Luna | text, image, file | yes | 1,050,000 | `$0.10 / $0.60` |
| Qwen 3.7 Plus | text, image | yes | 1,000,000 | `$0.32 / $1.28` |
| DeepSeek V4 Flash | text only | yes | 1,048,576 | `$0.0882 / $0.1764` |

DeepSeek cannot execute the multimodal Bongard interface. Qwen is a viable
visual fallback but costs 3.2 times Luna on input and about 2.13 times on
output. Generic benchmark strength does not replace a task-specific semantic
belief gate.

## Execution Boundary

The current exact-10, mechanics, development, and confirmation implementations
and manifests all bind Luna. No Qwen Bongard executor exists. Therefore the
preregistration sentence naming Qwen as the escalation model does not authorize
changing `MODEL_ID` inside a Luna run or reusing Luna-bound downstream
artifacts.

- If Luna exact-10 fails, bank the failure and make no mechanics call.
- If Luna mechanics fails, bank the failure and open no development task.
- In either failure case, a Qwen successor requires a separately frozen
  serving/mechanics implementation, costs, dates, seeds, manifests, and fresh
  downstream development/confirmation protocol before any affected response.
- If Luna mechanics passes, retain Luna for all four development blocks and
  conditionally authorized confirmation. Do not run Qwen as an unregistered
  parallel comparison.

## Budget

The current August 7 ledger records `$2.815715891` against the account-wide
`$5` cap, leaving `$2.184284109`. Live balance is `$24.886393846`. No paid
model-selection call is justified today: the official mechanics tasks are
date-frozen until August 10, while synthetic visual tasks would not authorize a
model change. This audit made zero model calls and spent `$0`.

The fresh August 10 preflight remains `ready_without_paid_calls`, with zero
files written, exact development-manifest SHA-256
`8659fb5fc6a02ddc59eb7147b6663d1fef3f96880e29f5e6de9bc0386f8e24aa`,
Luna live in the required modalities, and a `$2.00` maximum component-cap sum.
