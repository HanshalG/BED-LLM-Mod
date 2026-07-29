# ICAE Semantic-Matcher Exact-Response Controller Smoke

Date: 2026-07-29

## Rationale

The official-style ICAE Oracle smoke failed because one relevant question
missed its trigger and an identical trigger classification produced different
natural-language replies across fresh sessions. This is a scientifically
distinct controller, not a rerun or parser repair:

1. an LLM sees only released constraint IDs and trigger phrases and performs
   the irreducible free-form semantic match;
2. code returns the exact stored released response for matched IDs; and
3. unmatched questions return the exact stored fallback.

The matcher never sees stored responses. This preserves the LLM-native
semantic interface while making observations deterministic after matching.

## Frozen Tasks And Models

Exclude the two mechanics tasks opened by the failed Oracle smoke. Order the
remaining ten mechanics aliases by `SHA256("50400:<alias>")`; use the first
two:

- `realcode@235` (Kotlin);
- `realcode@185` (Go).

Models:

- support/question planner: `openai/gpt-5.4`, seed `50500`;
- semantic trigger matcher: `openai/gpt-5.4-mini`, seed `50600`.

Both run non-thinking at temperature zero.

## Exact Ten Calls

- two initial 12-hypothesis/six-question planner calls;
- two semantic matches of each first generated question;
- two exact repeated semantic matches on the same inputs;
- two semantic matches of the frozen unrelated generic question; and
- two history-conditioned support/question refreshes after the exact
  controller response.

No retries, repairs, normalization, reissue, task substitution, or model swap
is permitted.

## Gates

All must pass:

- exactly 10 accepted requests and 10 HTTP attempts;
- zero retries, reasoning tokens, and forced exits;
- strict parsing of all four planner supports and six matcher objects;
- both first questions match at least one released trigger;
- both repeated trigger-ID sets are exact;
- both generic questions return empty matches and exact fallback;
- controller output exactly equals stored response text by construction;
- at least four of six refreshed questions differ per task;
- each refreshed support uses at least one answer-introduced content term; and
- total cost is at most `$0.30`.

Passage authorizes only a separately frozen mechanics first-link experiment.
It does not authorize development tasks or executable efficacy.

## Accounting

- projected cost: `$0.12`;
- hard cap: `$0.30`;
- OpenRouter only;
- no OatML, Slurm, SSH, or cluster use.
