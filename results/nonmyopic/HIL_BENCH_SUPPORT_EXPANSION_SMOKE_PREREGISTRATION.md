# HiL-Bench Support-Expansion Smoke Preregistration

Frozen on 2026-07-25 before any HiL-Bench model response.

## Purpose

Test the first causal link required for LLM-native non-myopic BED: whether an
external observation causes the LLM to generate valid hidden-state support that
problem-only generation misses. This is not yet a policy comparison.

## Source And Cases

- Official HiL-Bench commit
  `352d14c861f2531949dfa91848d4b2fe46b8a247`.
- Development tasks `sql_48` and `sql_89`, fixed as the first two IDs in the
  source-clean seed-`24392` development split.
- Their blocker registries remain unloaded until every policy response is
  complete.
- GPT-5.4 through OpenRouter, temperature zero, explicitly non-thinking.
- No OatML cluster use.

## Exact Ten-Request Protocol

For each task:

1. One problem-only call generates four blocker hypotheses/questions and two
   business-document search queries.
2. One matched problem-only call regenerates four hypotheses without evidence.
3. Two calls regenerate four hypotheses after separate deterministic BM25
   top-five business-document observations.

These eight policy calls finish before hidden blocker data is loaded. Then one
post-freeze GPT-5.4 judge call per task matches all generated questions against
official blocker descriptions and example questions, without resolutions. The
judge cannot influence any policy response. Total: exactly ten physical
requests.

The matched problem-only regeneration is load-bearing: evidence receives no
extra generation calls. Search uses `rank-bm25 0.2.2` rather than the official
MiniLM retrieval server, so any result is a mechanism result on the released
documents, not an official HiL-Bench score.

## Frozen Gates

All must pass:

- exactly ten physical requests and HTTP attempts;
- zero retries, reasoning tokens, forced exits, repairs, or reissues;
- every initial, control, evidence-refresh, and judge object parses;
- the hidden registry is unavailable to all eight policy calls;
- the two retrieved observation lists differ on each task;
- all four generated supports are distinct on each task;
- on at least one task, the best evidence branch matches a blocker absent from
  both the initial and matched no-evidence supports;
- at least one such newly matched blocker has official type `business info`;
- total matched blockers from the per-task best evidence branches exceed both
  initial and no-evidence totals;
- adapter cost is at most `$0.75`.

Failure closes this exact smoke without prompt, parser, cohort, or threshold
repair. Passage authorizes a separately frozen paired planning experiment; it
does not itself establish a non-myopic advantage.

## Budget

Immediately before preregistration, the authenticated credits endpoint reported
`$43.475391484` remaining. Preserve `$25` through Monday. The smoke's hard cap
leaves at least `$17.725391484` above that reserve.
