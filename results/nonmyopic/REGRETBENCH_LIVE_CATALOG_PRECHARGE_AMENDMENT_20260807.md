# RegretBench Live-Catalog Precharge Amendment

Date: 2026-08-07

Status: prospectively amended before any RegretBench response. Model calls and
cost: `0` / `$0`.

## Problem

The Aug 8 Luna baseline wrapper already checked the live OpenRouter model
catalog before dispatch, but the downstream RegretBench wrappers checked only
the inherited account balance and local protocol hashes. A removed endpoint,
loss of seeded structured output, reduced context/completion limits, invalid
pricing, or price increase beyond the fixed per-request reservation could
therefore be discovered only during or after a paid request. With concurrency
up to `128`, an underestimated reservation is an account-wide budget risk.

## Amendment

Both RegretBench daily preflights now read the authenticated OpenRouter catalog
without making a model request. The support-recovery wrapper requires exactly
one `deepseek/deepseek-v4-flash-0731` record with:

- text input and output;
- seeded requests and structured output;
- context length at least `65,536` and completion limit at least `2,200`;
- finite nonnegative input/output prices; and
- a `$0.0015` attempt reservation that covers all `2,200` output tokens plus
  at least `4,096` prompt tokens at the live prices.

The policy wrapper repeats that DeepSeek check and also applies the existing
Luna multimodal/reasoning/structured-output and `$0.008` reservation check. A
DeepSeek failure aborts before any adapter construction, request, output
directory, or daily ledger. A Luna failure is instead banked as
`unavailable_preflight`: no Luna adapter or naive smoke is opened, while the
primary DeepSeek smoke and development remain authorized. This preserves the
preregistered rule that the descriptive baseline cannot veto primary science.

The `4,096` floor is tied to the frozen interface. Exhaustive construction over
all 64 development CIGs, zero/one/two dialogue rounds, maximum schema question
and generated-reply lengths, and the longest official slot value gives a
maximum serialized request of `3,354` UTF-8 bytes, including the response
schema and request controls. Token count cannot exceed the byte count, leaving
a conservative margin.

## Live Read-Only Result

At audit time, 0731 reports `$0.09/$0.18` per million input/output tokens,
1,048,576 context, 65,536 maximum completion tokens, seeded structured output,
and `12,266.67` prompt tokens covered after reserving the full 2,200-token
completion. Luna reports `$0.10/$0.60`, 128,000 maximum completion tokens, and
`30,848` prompt tokens covered by its existing reservation. The authenticated
credit endpoint remains `$245.000000000` credits, `$220.113606154` usage, and
`$24.886393846` balance; the reported `$30` top-up is still unposted.

## Scientific Invariance

This amendment changes no model, prompt, seed, response schema, task, split,
policy, endpoint, call count, concurrency, cap, threshold, or analysis. It adds
only a fail-before-dispatch operational gate. All scientific preregistration
and core hashes remain unchanged.
