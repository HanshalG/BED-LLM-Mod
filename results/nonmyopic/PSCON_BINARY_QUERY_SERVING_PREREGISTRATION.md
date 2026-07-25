# PSCon Role-Separated Binary Serving Gate

## Purpose

Test the distinct decomposition suggested by the closed joint three-way interface:

1. Mini generates one semantic yes/no clarification question.
2. A separate Mini call maps all candidate titles to `Y`, `N`, or `U`.

This is the minimal pair of LLM-native operations needed by a future semantic-tree
planner. The gate is target-free: no liked product, independent responder, policy
score, or endpoint is loaded.

## Frozen Protocol

- PSCon source and hashes: identical to the pinned prior gates.
- Input: already-open conversation `64937`, visible history, and 20 product titles.
- Seed: `24387`.
- Model: `openai/gpt-5.4-mini`, explicitly non-thinking.
- Five question calls at temperature `.7`.
- Five classification calls at temperature `0`.
- Exactly 10 physical calls, concurrency 5.
- Cost cap `$0.15`; projected cost `$0.03`.
- Raw outputs checkpoint before parsing.
- No normalization, repair, continuation, reissue, target access, or OatML.

Questions must be one bounded line ending in `?` or `？`. Each classification must be
exactly 20 characters from `Y/N/U`, one per product in source order.

## Frozen Gates

All must pass:

- exactly 10 requests and HTTP attempts;
- zero retries, reasoning tokens, and forced exits;
- all five questions and all five label strings parse;
- at least four unique questions;
- at least four unique likelihood partitions;
- at least four partitions use two or more labels and have entropy at least `.30`
  nats;
- informative-partition entropy range at least `.10` nats;
- cost at most `$0.15`.

Failure closes this role-separated Mini interface and the current PSCon semantic-tree
route; there is no serving V2. Passage authorizes only a separately preregistered
efficacy run on a different English development case, with independent GPT-5.4
responses and compute-matched myopic/width/random controls. Chinese data remains
sealed.

## Dry Verification

Before real calls:

- 14 focused PSCon tests pass;
- compilation and `git diff --check` pass;
- the full deterministic source-backed fixture passes every gate with five unique,
  informative partitions and `.5432` nats of entropy range.
