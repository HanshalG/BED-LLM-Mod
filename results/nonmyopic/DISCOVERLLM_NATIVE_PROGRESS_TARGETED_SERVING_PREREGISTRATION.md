# DiscoverLLM Native-Progress Targeted Serving Preregistration

Frozen after closing the generic native-progress serving task and before
loading this fresh task or obtaining a targeted-interface response.

## Distinct Design

The generic action bank produced zero root probes and zero root advances. This
successor changes the experimental design, not the failed parser:

- `D2` explicitly contrasts all four candidate current priorities, creating a
  high-immediate-information dialog action that cannot advance.
- `R2` contains exactly four LLM-written artifact alternatives, one designed
  to fully satisfy each candidate priority subtree. DiscoverLLM's native
  best-alternative rule permits different alternatives to satisfy different
  nodes, so this action can advance the hidden root without knowing truth.

`D1` remains a broad clarification and `R1` a common-ground artifact synthesis.
All four actions are shared across worlds.

The transition codec is also scientifically coarsened to the state variables
that affect planning:

- `A`: artifact fully satisfies every leaf and advances;
- `C`: root stays active and the user's priority becomes clear;
- `V`: root stays active and remains vague;
- `T`: already terminal.

This prevents irrelevant dialog/artifact outcome combinations while preserving
the official causal rule: only `A` advances.

## Frozen Source And Split

- Official code commit: `a9eb2846f60e3681ac8d325fc57fd4e58e2bdc97`.
- V2 manifest SHA-256:
  `9edfd3b20f762491db78087c95bccb1d345063af3423d4d7ccf6c481aa97ad3a`.
- Serving task: `technical_writing:artifact_352`.
- Reserved mechanics tasks:
  `technical_writing:artifact_83`,
  `creative_writing:artifact_348`,
  `svg_drawing:artifact_27`.
- Generic native-progress and all earlier artifacts are excluded.
- Opportunity 60 and holdout 162 remain sealed.

Worlds remain the first four source-ordered eligible current-root positions.
World presentation uses seed `24418`; observation shuffling remains seed
`24417`. Tier weights remain prospectively fixed at `H/M/L = 4/2/1`.

## Serving Gate

Run the same eight semantic stages as V1 with GPT-5.4, temperature zero, no
reasoning, and the new action/state interfaces. Pass requires:

- every action, transition, feedback, likelihood, and follow-up cell parses
  exactly;
- `D1` and `D2` never advance;
- targeted `R2` advances at least two of four candidate worlds;
- exactly eight logical requests, at most three logged transport retries, HTTP
  attempts equal requests plus retries, zero forced exits/reasoning tokens;
- no semantic retry, repair, coercion, or partial analysis; and
- cost at most `$0.35` (projected `$0.22`).

Passing authorizes only a separately frozen three-task mechanics smoke.
Failure closes this exact targeted design and task without another action or
codec variant.

OpenRouter only. OatML jobs: `0`.
