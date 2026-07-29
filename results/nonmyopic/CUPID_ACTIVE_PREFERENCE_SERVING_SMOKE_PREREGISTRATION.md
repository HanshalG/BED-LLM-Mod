# CUPID Active-Preference Serving Smoke Preregistration

Date frozen: 2026-07-29

Status: **frozen before any planner or target response**.

## Purpose

Test whether two independent non-thinking model endpoints can serve the minimum
semantic objects required for active, non-myopic contextual-preference BED:

1. an LLM-generated open-text support with useful binary clarification
   partitions; and
2. user answers grounded in CUPID's hidden released preference that remain
   represented by that support.

This is a serving and opportunity gate only. It does not select or evaluate a
policy and makes no depth claim.

## Bound Source And Cases

- Source manifest SHA-256:
  `2f742ddbacd64eade99fa0148b3c5a11a8676f5cfceb5e666f0966fa6785f790`.
- Use exactly the five `serving_smoke` rows listed in that manifest: one
  `consistent`, two `contrastive`, and two `changing`.
- The planner sees only the manifest-bound candidate payload: current
  request/context and two nonmatching-context prior dialogues.
- The planner never receives the released current preference, checklist,
  structured prior preference metadata, same-context dialogue, or remaining
  histories.
- The target sees the current request/context, released current contextual
  preference, and generated questions. It receives no checklist or prior
  dialogue.
- No development or holdout row is accessed.

## Frozen Calls

- Planner: `openai/gpt-5.4-mini`, non-thinking, seed `37500`,
  temperature `.7`.
- Target: `google/gemini-2.5-flash`, non-thinking, seed `37600`,
  temperature `0`.
- Exactly five planner calls followed by five target calls.
- Concurrency `5` within each phase.
- Zero retries, repairs, continuations, reissues, or substituted models.
- Structured JSON output on both phases.
- Planner output per case:
  - exactly six unique yes/no questions `Q1`--`Q6`;
  - exactly twelve open-text preference hypotheses `H1`--`H12`; and
  - one exact six-bit predicted answer signature per hypothesis.
- Target output per case: one exact six-bit answer signature.
- Planner maximum output 6,000 tokens; target maximum output 128 tokens.
- Projected cost `$0.10`; hard cap `$0.25`.
- OpenRouter only; no OatML.
- Raw prompts/responses and hidden preferences remain private. Public output
  contains source IDs, counts, entropy/coverage diagnostics, usage, and a
  private-artifact hash only.

## Frozen Gates

All gates are conjunctive:

- exactly 10 accepted requests and exactly 10 HTTP attempts;
- zero retries, provider-error retries, reasoning tokens, and forced exits;
- all five planner and target responses parse under the exact schemas;
- every case has six unique questions;
- every case has at least 10 unique normalized preference hypotheses;
- every case has at least 8 unique hypothesis answer signatures;
- each of the 30 question partitions has a minority side of at least 2 of 12
  hypotheses;
- every case has mean uniform-support partition entropy at least `.45` nats;
- the independent target's exact answer signature occurs in the generated
  support on at least 4 of 5 cases;
- the nearest generated signature is within Hamming distance 1 on all 5 cases;
- target responses produce at least 3 distinct signatures across the 5 cases;
  and
- total reported cost is at most `$0.25`.

Failure closes this exact planner/target interface. There is no threshold
repair, row removal, prompt revision, model swap, retry, or same-case successor
after viewing aggregate responses.

Passage authorizes only a separately specified development mechanics gate. That
gate must establish path-dependent support regeneration and an independently
scored preference/checklist endpoint before any paired non-myopic policy test.

## Dry Verification

Before real calls, the deterministic source-backed fixture must complete the
same two-phase control flow with exactly 10 accepted requests and pass every
gate. Focused source and serving tests must pass. The implementation and this
preregistration must be committed and pushed before execution.
