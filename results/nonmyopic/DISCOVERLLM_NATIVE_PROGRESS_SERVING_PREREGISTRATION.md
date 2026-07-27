# DiscoverLLM Native-Progress Serving Preregistration

Frozen before loading the designated native-progress artifact or obtaining any
response under this interface.

## Motivation And Claim Boundary

The released two-action priority-world construction failed because both source
completions elicited identical root information on the two changed tasks. This
successor is scientifically distinct: the LLM generates a shared semantic
action bank, and transition dynamics follow DiscoverLLM's released native state
machine.

The official simulator distinguishes:

- a **dialog act**, which may probe and increase awareness but does not satisfy
  or advance the current criterion root; and
- an **artifact**, which advances only when every leaf in the current root is
  satisfied.

That creates a genuine enabling mechanism. A dialog question can be
immediately diagnostic, while an artifact action can sacrifice immediate
information to unlock feedback about the next hidden root.

This one-task run tests only realistic serving, strict parsing, and native state
execution. It is not a mechanics, policy, opportunity, or endpoint result.

## Frozen Source And Split

- DiscoverLLM code commit:
  `a9eb2846f60e3681ac8d325fc57fd4e58e2bdc97`.
- V2 manifest SHA-256:
  `9edfd3b20f762491db78087c95bccb1d345063af3423d4d7ccf6c481aa97ad3a`.
- Serving task: `creative_writing:artifact_305`.
- Reserved mechanics tasks:
  `creative_writing:artifact_100`,
  `technical_writing:artifact_37`,
  `technical_writing:artifact_359`.
- Every prior serving/mechanics artifact is excluded.
- All 60 opportunity and 162 holdout artifacts remain sealed.
- Released scores and winner labels are never read.

For each task, source-ordered eligible roots define four equally likely hidden
starting positions. World `Wi` starts at ordered root `i`; successful artifact
actions advance to the next source root. No root text or order is invented.

## Exact Eight-Stage Interface

Use `openai/gpt-5.4`, temperature zero, without reasoning:

1. Generate four shared actions: broad dialog `D1`, contrastive dialog `D2`,
   single artifact `R1`, and multiple artifact alternatives `R2`.
2. Apply the official dialog/artifact probing/satisfaction transition semantics
   to every action/world cell.
3. Generate natural world-conditioned root feedback from the resulting active
   state.
4. Assign coarse `H/M/L` likelihood tiers to starting worlds.
5. Generate one target-blind continuation for every observed branch.
6. Apply the same native transition semantics to every branch/world cell.
7. Generate world-conditioned follow-up feedback from the new active state.
8. Assign terminal `H/M/L` likelihood tiers to starting worlds.

World presentation is shuffled with seed `24416`; root observations are
shuffled with seed `24417`. The likelihood scorer sees candidate transition
trajectories but never the observation-to-truth map. The continuation policy
sees only visible branch history.

## Serving Gate

Pass requires:

- exact parsing of 4 actions, 16 root transitions, 16 root feedback cells,
  16 root tier rows, 16 continuations, 64 follow-up transitions, 16 follow-up
  feedback cells, and 16 terminal tier rows;
- generated `D1/D2` cells classify as dialog and `R1/R2` cells as artifact;
- exactly eight logical requests, at most three logged transport retries, and
  HTTP attempts equal requests plus retries;
- no semantic repair, coercion, reissue, or partial analysis;
- zero reasoning tokens and forced exits; and
- cost at most `$0.30` (projected `$0.18`).

Passing authorizes only a separately frozen three-task mechanics smoke.
Failure closes this exact interface and task without a format-only rerun.

OpenRouter only. OatML jobs: `0`.
