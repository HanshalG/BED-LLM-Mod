# HiL-Bench Progressive-Support Opportunity Preregistration

Frozen before running the opportunity audit on 2026-07-25. This is a zero-call
structural gate. It cannot establish policy efficacy.

## Source And Split

- Official source: `https://github.com/hilbenchauthors/hil-bench`
- Pinned commit: `352d14c861f2531949dfa91848d4b2fe46b8a247`
- Domain: the 100 released SQL tasks.
- `sql_0` through `sql_10` are excluded because their task-level source was
  inspected while validating the release.
- Seed `24392` shuffles `sql_11` through `sql_99` before any remaining
  task-level blocker outcome is read.
- The first 40 tasks are the opportunity cohort, the next 20 are development,
  and the final 29 are sealed holdout.

The exact IDs and split-reproduction check live in
`scripts/hil_bench_progressive_opportunity.py`.

## Proposed Mechanism

The hidden state is the official blocker registry. The policy sees the natural
SQL request but never sees blocker descriptions, example questions, or
resolutions. Its belief support is instead an LLM-generated set of possible
blockers and targeted clarification questions.

An inspection action such as `get_business_info` has no immediate clarification
reward. Its observation can nevertheless cause the LLM to generate a valid
blocker hypothesis and question that was absent initially. A depth-two policy
can value that path-dependent support expansion; one-step EIG on the current
generated support cannot. This is the specific LLM-native non-myopic mechanism
under test.

Blockers are independent in the released benchmark. Therefore source diversity
or observation relevance alone is not a greedy-planning result. A pass below
authorizes only a small model-serving and support-expansion smoke.

## Frozen Audit

For each opportunity task, verify the official task shape and count blocker
source annotations. For each blocker annotated `business info`, tokenize its
description, the initial request, and every released business-information
document. Measure whether at least one document contains blocker-description
content words absent from the initial request. The audit records counts only;
no blocker text, example question, or resolution is copied into its artifact.

All of these gates must pass:

| Gate | Threshold |
| --- | ---: |
| usable opportunity tasks | exactly 40 |
| blocker count | 3--5 on every task |
| tasks with both question and business-info blockers | at least 34 |
| tasks with question, business-info, and schema blockers | at least 30 |
| business-info blockers | at least 40 |
| business-info blockers with novel observed evidence | at least 60% |
| mean novel blocker-token recall from the best business document | at least 0.03 |

## Consequence

- **Pass:** preregister and run a deterministic fixture test, followed by at
  most ten GPT-5.4 non-reasoning policy calls on development tasks. OpenRouter
  balance must still preserve a hard `$25` reserve through Monday.
- **Fail:** close HiL-Bench without model calls or threshold/cohort tuning.
- No OatML cluster work is authorized.
