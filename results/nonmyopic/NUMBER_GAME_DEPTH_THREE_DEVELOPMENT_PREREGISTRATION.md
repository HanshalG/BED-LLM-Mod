# Number Game Depth-Three Development Preregistration

Date frozen: 2026-07-28, before any eight-tree development response.

This is a development gate for a genuine three-query planner, not a
confirmation. Two excluded one-tree mechanics smokes were inspected first.

## Excluded Smokes

- Seed `27390/27490`: 50/50 calls, no retry/reasoning/forced exit,
  first-branch minimum 11, second-branch minimum 3, cost `$0.1236533`;
  depth-three and depth-two roots both 22.
- After a prompt-only hard-constraint clarification, seed `27391/27491`:
  50 accepted / 52 attempts, two explicit zero-cost provider-error retries,
  no reasoning/forced exit, first-branch minimum 11, second-branch minimum 4,
  cost `$0.1302163`; depth-three root 80 versus depth-two root 84.

Only mechanics, costs, and source-policy root identities were inspected.
Independent target endpoint metrics are excluded.

## Frozen Development

- Eight fresh Gemini planning trees, seeds `27400..27407`.
- Eight fresh GPT-5.4 target supports, seeds `27500..27507`.
- Nonreasoning, temperature `0.7`, strict 24-rule grammar.
- Per tree: one initial support; 16 supports after each root answer; an
  adaptive greedy second query in each first branch; 32 supports after both
  possible second answers; one independent target support. Exactly 50
  successful responses per tree, 400 total.
- The depth-three first root minimizes terminal predictive Brier after a
  greedy second query, a second LLM support refresh, and a greedy third query.
- Controls execute the same three-query continuation after roots selected by
  depth-two predictive risk, myopic EIG, exact static-support depth three,
  seeded PTS, and exact uniform random.
- Minimum valid supports: 16 initial, eight first-step, four second-step,
  16 targets, and eight targets novel to initial support.
- Total cap: `$1.20`; no reserve.

## Promotion Rule

A fresh confirmation is allowed only if all mechanics gates pass and, versus
depth-two predictive risk:

1. the selected first root differs on at least 4/8 trees;
2. aggregate Brier improves by at least 3%;
3. Brier improves on at least 5/8 tree means;
4. mean Hamming does not increase; and
5. mean exact-extension coverage does not decrease.

Depth three must also have directional Brier gains over myopic EIG and exact
uniform random. Failure closes this implementation of a third planning step;
thresholds, trees, and branch supports will not be repaired after endpoints.
