# RegretBench Model Allocation Audit

Date: 2026-08-07

Status: zero-call allocation decision. It does not alter the frozen August 8
RegretBench chain or authorize a model substitution.

## Question

Under a hard account-wide `$5.00` Europe/London daily ceiling, should the
LLM-native BED program use GPT-5.6 Luna, DeepSeek V4 Flash 0731, or both?

The relevant comparison is role-specific. Generic reasoning scores do not
measure the nonreasoning structured particle generator used by the planner,
and a text-only model cannot serve the Bongard image environment.

## Current External Evidence

Artificial Analysis reports GPT-5.6 Luna at high reasoning effort with an
Intelligence Index of `47`, image and text input, 37M benchmark output tokens,
and first-party prices of `$0.20/$1.20` per million input/output tokens. The
same page describes a 1M-token context and 175 output tokens/second:

https://artificialanalysis.ai/models/gpt-5-6-luna-high

Artificial Analysis reports DeepSeek V4 Flash **nonreasoning** at an estimated
Intelligence Index of `29`, text-only input, and first-party prices of
`$0.14/$0.28` per million input/output tokens:

https://artificialanalysis.ai/models/deepseek-v4-flash-non-reasoning

The higher DeepSeek numbers are reasoning-mode results and are not applicable
to the planner role. The original max-reasoning V4 Flash result is `40` in the
current model page. Its April release analysis also reports very high output
token use and a 96% hallucination rate on AA-Omniscience:

https://artificialanalysis.ai/models/deepseek-v4-flash

https://artificialanalysis.ai/articles/deepseek-is-back-among-the-leading-open-weights-models-with-v4-pro-and-v4-flash

Authenticated OpenRouter catalog inspection on 2026-08-07 gives the exact
deployed contracts:

| Model | Input / 1M | Output / 1M | Input modes | Frozen role |
|---|---:|---:|---|---|
| `openai/gpt-5.6-luna` | `$0.10` | `$0.60` | text, image, file | multimodal belief work; labelled medium-reasoning naive baseline |
| `deepseek/deepseek-v4-flash-0731` | `$0.09` | `$0.18` | text only | high-volume nonreasoning semantic support and planning |

At these live prices Luna is only `1.11x` DeepSeek on input but `3.33x` on
output. Completion volume therefore determines the cost difference. The
unpinned `~deepseek/deepseek-v4-flash-latest` alias is not an experimental
substitute for the frozen `0731` endpoint.

## Direct Interface Evidence

Our measurements dominate the generic indexes for this project:

- Luna's Number Game reliability gate produced four malformed/forced outputs
  in 128 requests. It is not transport-reliable enough to replace the
  high-volume structured planner without a new interface gate.
- DeepSeek 0731 completed 128/128 strict structured requests with zero parse,
  retry, provider, reasoning, or forced-exit failures. Its failure was semantic:
  too many conditioned draws had fewer than four valid hypotheses.
- A second DeepSeek diversity prompt reduced duplicate and schema failures but
  worsened observation inconsistency. Generic intelligence did not repair that
  support contract.
- Neither result has opened the RegretBench endpoint. The frozen RegretBench
  exact-10 support smoke is the correct task-specific decision gate.

## Allocation Decision

Keep the August 8 chain unchanged:

1. Luna runs only the exact-10 medium-reasoning naive baseline smoke.
2. DeepSeek 0731 runs the text-only support-recovery gate.
3. The high-volume dynamic policy opens only after a literal independently
   verified DeepSeek support pass.
4. Any null stops descendants and banks the unused daily allowance; it does
   not trigger an unregistered same-day model swap.

For Bongard, Luna is mandatory because the live DeepSeek endpoint is text-only.
For a future factorized RegretBench policy, DeepSeek remains the first static
likelihood annotator because the task is simpler than semantic transition and
the output-heavy call count makes Luna materially more expensive. A Luna
transition draw is a separately preregistered future comparison only after
completed-history residual evidence, never an automatic rescue.

## Budget Rule

The hard limit is `$5.00` of account-wide spend per London calendar day,
including unrelated usage and all in-flight reservations. The August 8
RegretBench chain has a frozen `$4.80` maximum, leaving `$0.20` scheduling
slack. It spends less when an early gate is null. Unused allowance does not
roll over.

Authenticated credits remain `245.000000000`, usage `220.113606154`, and
balance `$24.886393846`. The user-reported `$30` top-up is not budgeted until
the provider's `/credits` endpoint posts it. Once posted, it supplies six more
daily ceilings, not one `$30` day.

## Conclusion

Luna is the stronger and uniquely multimodal model; DeepSeek 0731 is the
economically feasible high-volume text planner. Neither is a universal default.
The Pareto-optimal allocation is Luna for the irreducible image/reasoning roles
and DeepSeek for nonreasoning structured volume, with task-specific exact-call
gates overriding AA Index position.
