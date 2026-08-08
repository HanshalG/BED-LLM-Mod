# OpenRouter Model Pareto And Daily Route

Date: 2026-08-08 (Europe/London)

Status: zero-call routing audit. This supersedes the generic future-model advice in
`OPENROUTER_BUDGET_MODEL_ROUTE_20260808.md`; it does not alter any frozen model,
prompt, effort, task, seed, endpoint, threshold, request count, or cost cap.

## Account Boundary

The authenticated OpenRouter account currently reports:

- total credits: `$245.000000000`;
- cumulative usage: `$220.121013787`;
- available balance: `$24.878986213`;
- current key daily usage: `$0.007407633`.

The newly reported `$30` top-up is not yet present in `/api/v1/credits`, so it is not
available to a ledger yet. If it posts without intervening use, the balance will be
`$54.878986213`. The operating boundary remains a hard account-wide `$5.00` cap per
London calendar day. The cap includes unrelated account use and all full worst-case
HTTP-attempt reservations. Unused allowance neither rolls over nor justifies inventing
an invalid experiment.

The daily priority is:

1. the strongest dependency-valid headline stage;
2. its preregistered paired baseline or matched-compute control;
3. an exact task-interface model gate needed by the next stage;
4. bounded transport headroom.

This is how to use `$5` properly: spend toward the active claim until a registered
gate closes the branch. A null can therefore make a scientifically correct day cost
less than `$5`.

## Current Intelligence-Cost Frontier

Artificial Analysis v4.1.1 evaluates reasoning efforts separately. The total-eval
cost is more useful than token list price alone because reasoning verbosity differs
substantially.

| Model and effort | AA Index | Evaluation output | Full AA eval cost | Input modality |
|---|---:|---:|---:|---|
| GPT-5.6 Luna, nonreasoning | 27 | 2.4M | `$10.21` | text, image |
| GPT-5.6 Luna, medium | 39 | 12M | `$21.15` | text, image |
| GPT-5.6 Luna, high | 47 | 37M | `$55.02` | text, image |
| GPT-5.6 Luna, max | 52 | 130M | `$172.17` | text, image |
| DeepSeek V4 Flash 0731, max | 52 | 210M | `$72.03` | text only |

These are first-party evaluation costs, not our OpenRouter bill. They show the shape
of the quality curve:

- For hard text reasoning, DeepSeek 0731 max reaches the same AA score as Luna max at
  less than half the full-evaluation cost. It is the text-only frontier model of this
  pair, despite being much more verbose.
- For visual reasoning, Luna is the only applicable model. Luna high is the sensible
  escalation point: it gains eight AA points over medium, while max adds only five
  more points for over three times the full-evaluation cost.
- For high-volume structured generation, max-reasoning scores are not transferable.
  The deployed nonreasoning interface must pass its own schema, semantic-support,
  and ranking-fidelity gates.

The authenticated OpenRouter catalog currently prices the exact routes as follows:

| Exact route | Input / output per 1M | Context | Capabilities |
|---|---:|---:|---|
| `deepseek/deepseek-v4-flash-0731` | `$0.0896 / $0.1792` | 1,048,576 | text, reasoning effort, structured output |
| `openai/gpt-5.6-luna` | `$0.10 / $0.60` | 1,050,000 | text, image, file, reasoning effort, structured output |

Use the dated DeepSeek route, never the mutable `~...-latest` alias. DeepSeek must
never receive Bongard images.

## Project-Specific Evidence

Direct task evidence overrides the generic frontier:

- DeepSeek 0731 completed `1,568` nonreasoning Number Game planner calls for
  `$0.3496433702`. Its depth-three Brier was `4.0619%` below its own myopic policy,
  but it missed the registered mechanics and efficacy gates.
- In the conditioned-support reliability test, DeepSeek parsed `128/128` strict
  responses but failed the semantic support floor. Cheap, valid JSON was not enough.
- Luna's scaled Number Game reliability test had four malformed or forced outputs.
  It is not a drop-in high-volume text planner without a fresh exact-interface gate.
- Luna medium-reasoning passed the current multimodal Bongard naive exact-10 smoke:
  `10/10` accepted, zero retries/errors/forced exits, cost `$0.006675400`.
- The DeepSeek RegretBench smoke stopped on a bounded `IncompleteRead` transport
  failure after three accepted responses. That is infrastructure evidence, not a
  semantic null and not authorization to rerun.

## Frozen Routing Policy

1. **Bulk text-only support, likelihood, and planning:** DeepSeek 0731
   nonreasoning, after an exact schema and semantic gate.
2. **Small hard text reasoning gate:** DeepSeek 0731 high or max, with a fixed output
   cap and observed cost projection before scale.
3. **Bulk visual semantic belief:** Luna nonreasoning. This is the frozen Bongard
   policy model and remains unchanged.
4. **Thinking/naive visual baseline:** Luna medium, explicitly labelled as reasoning.
5. **Visual quality escalation:** Luna high first. Luna max is considered only if a
   high-effort exact task gate shows a material shortfall that max could address.

No reasoning effort is silently enabled for the planner or environment roles. No
model is swapped after looking at a scientific endpoint. The unopened Bongard chain
therefore remains exactly Luna nonreasoning for belief dynamics plus the separate
Luna-medium naive baseline.

## Sources

- Artificial Analysis, GPT-5.6 Luna nonreasoning:
  https://artificialanalysis.ai/models/gpt-5-6-luna-non-reasoning
- Artificial Analysis, GPT-5.6 Luna medium:
  https://artificialanalysis.ai/models/gpt-5-6-luna-medium
- Artificial Analysis, GPT-5.6 Luna high:
  https://artificialanalysis.ai/models/gpt-5-6-luna-high
- Artificial Analysis, GPT-5.6 Luna max:
  https://artificialanalysis.ai/models/gpt-5-6-luna
- Artificial Analysis, DeepSeek V4 Flash 0731 max:
  https://artificialanalysis.ai/models/deepseek-v4-flash
- OpenRouter, DeepSeek V4 Flash 0731:
  https://openrouter.ai/deepseek/deepseek-v4-flash-0731

This audit made zero paid model calls and changed no paid execution path.
