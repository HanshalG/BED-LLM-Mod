# CUPID Active-Preference Serving Smoke V2 Preregistration

Date frozen: 2026-07-29

Status: **frozen before any V2 model response**.

## Rationale

V1 failed after five planner calls and before hidden-target access because three
responses inserted separators into six-character answer strings. V1 remains
failed. V2 is a disjoint serving qualification that changes only the JSON
representation of each six-bit signature to an array of six integers whose
items are schema-constrained to `0` or `1`.

V2 does not reuse a V1 row or response. It keeps V1's models, prompts,
two-phase control flow, scientific gates, concurrency, and cost cap.

## Bound Cases

Use exactly these previously unopened rows from the source manifest's
development split, in order:

1. `79+mathematics_professor:consistent`
2. `66+horticulturist:consistent`
3. `85+design_research_associate:contrastive`
4. `212+maritime_preservationist:contrastive`
5. `153+music_therapist:changing`

Source manifest SHA-256:
`2f742ddbacd64eade99fa0148b3c5a11a8676f5cfceb5e666f0966fa6785f790`.

V1 public failure SHA-256:
`a65e45184c26e1380e113fbc99088ad99741cbc1cb727e5c7cdfa2a4f5ed7e71`.

The same hidden-state boundary applies. The planner sees current
request/context plus two nonmatching-context dialogues. The target, conditional
on a valid planner phase, sees current request/context, the released hidden
preference, and six generated questions. Neither model sees the checklist.

## Frozen Calls

- Planner: `openai/gpt-5.4-mini`, non-thinking, seed `37510`,
  temperature `.7`.
- Target: `google/gemini-2.5-flash`, non-thinking, seed `37610`,
  temperature `0`.
- Exactly five concurrent planner calls, then five concurrent target calls.
- Exact V1 prompts, including `1` for yes and `0` for no.
- Structured V2 signatures are arrays of exactly six integer `0/1` items.
- No `uniqueItems`, parser repair, response normalization beyond joining six
  already-valid integers, retry, continuation, reissue, or model substitution.
- Planner maximum output 6,000 tokens; target maximum output 128 tokens.
- Projected cost `$0.10`; hard cap `$0.25`.
- Raw data private; public artifact contains metrics, usage, IDs, and hashes.
- No OatML, checklist endpoint, policy endpoint, or holdout access.

## Frozen Gates

The V1 gates are copied without threshold changes. All are conjunctive:

- exactly 10 accepted requests and 10 HTTP attempts;
- zero retries, provider-error retries, reasoning tokens, and forced exits;
- all five planner and target responses parse exactly;
- every case has six unique questions;
- every case has at least 10 unique normalized hypotheses;
- every case has at least 8 unique hypothesis signatures;
- every question has a minority partition of at least 2 of 12;
- every case has mean partition entropy at least `.45` nats;
- exact target-signature coverage on at least 4 of 5 cases;
- nearest target signature within Hamming distance 1 on all 5 cases;
- at least 3 distinct target signatures across cases; and
- total cost at most `$0.25`.

Failure closes CUPID serving for this project; no V3, row removal, threshold
change, prompt revision, representation change, or model swap follows.

Passage authorizes only a separately preregistered development mechanics gate
for path-dependent support regeneration and an external preference/checklist
endpoint. It does not authorize a policy claim by itself.

## Dry Verification

Before any real V2 response, focused V1/V2/source tests and the exact ten-call
deterministic V2 fixture must pass. This document, implementation, and tests
must be committed and pushed before execution.
