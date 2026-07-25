# tau-Knowledge Retrieval Opportunity V2

## Status

V1 passed its two-task smoke but failed closed on one malformed nested followup
before opportunity endpoints. V2 is a distinct serving interface and fresh
mechanism split. It does not reuse, repair, or reissue the V1 response.

## Frozen changes

- Source repository and commit remain
  `sierra-research/tau2-bench@1d244f5dca42944b67a379b44bfeb9f5748f189d`.
- Selection seed: `24335`.
- V2 eligibility is the 30 official tasks without the exact scripted-opening
  pattern, excluding inspected `task_001` and `task_026`.
- Smoke: `task_002`, `task_024`.
- Opportunity: `task_018`, `task_008`, `task_014`, `task_010`, `task_006`,
  `task_003`.
- The original 20 V1 confirmation tasks remain sealed and unchanged.
- One GPT-5.4 non-reasoning call converts each private official user script to a
  single concise first utterance. It is forbidden to expose verification data,
  later steps, hidden actions, policy solutions, or evaluation criteria.
- The query planner receives only this utterance, never the private script.
- Initial and followup generation use flat numbered string keys:
  `need_1` through `need_8`, plus `direct_query_1/2`,
  `enabling_query_1/2/3`, or `followup_query_1` through `4`. Arrays and nested
  objects are forbidden.

Everything scientific is unchanged from V1: GPT-5.4 non-reasoning,
temperature zero, eight path-dependent information-need hypotheses, five first
queries, four followups per root, official `rank-bm25==0.2.2` top-3 retrieval,
exact annotated required-document endpoints, oracle-strength greedy first
control, and all opportunity thresholds.

## Gates

Smoke is exactly 14 physical calls: two openings, two initial plans, and ten
followups. It requires complete flat schemas and trees, zero reasoning, exact
request count, and at least three distinct first top-1 documents per task.

Conditional opportunity is exactly 42 calls. It passes only with:

- mean first top-1 diversity >=3;
- oracle pair retrieves required policy on >=4/6 tasks;
- pair gain >=1 document on >=3/6;
- oracle/greedy root differs on >=2/6;
- non-myopic gap >=1 document on >=2/6;
- mean pair gain >=0.50; and
- mean non-myopic gap >=0.33.

Passing authorizes a separate target-blind scorer preregistration on the sealed
20 tasks. Failure closes V2. No response repair, replacement, task substitution,
or threshold change is allowed.

## Budget

Before V2, project headroom was `$34.66230053` and live balance above the `$25`
Monday reserve was `$34.66230054`. OatML remains paused. Projected smoke and
opportunity costs are `$0.20/$0.70`, with hard caps `$0.75/$2`.
