# LogDx-CI Semantic BED Serving Smoke Preregistration

Date: 2026-07-29

Status: **frozen before any serving response**.

## Authorization And Scope

The corrected LogDx-CI source audit passed every frozen gate:

- source audit V3 SHA-256:
  `7f6666a19b635087b78015a1ff98b8082c897c60a2eaca20ab8e73440931c4ff`;
- official LogDx-CI v1.2 commit:
  `99591c1471118c95155976346df72f520a05f100`.

This smoke qualifies only the semantic interface required for BED. It does not
select a policy, score a diagnosis, read `ground_truth.json`, or use any
confirmation case.

## Fixed Development Cases

Take the first five of the eight legacy/v2 development IDs after sorting by
`sha256("37800|" + case_id)`:

1. `pip-pytest-network-github-v2-001`
2. `mypy-pandas-001`
3. `pytest-pandas-001`
4. `cargo-tokio-001`
5. `pnpm-jest-config-v2-001`

The assistant-visible initial context is the released `rtk-log` reduction for
each case plus safe metadata (`case_id`, repository/source, workflow, job, and
framework). Raw logs are accessible only through the released deterministic
tools.

## Exact Calls

Phase one, five concurrent calls:

- model: `openai/gpt-5.4-mini`;
- non-thinking;
- seed `37900`;
- temperature `.7`;
- maximum output `3,000` tokens.

Each response must contain exactly:

- six ordered, distinct semantic root-cause hypotheses `H1` through `H6`;
- four ordered, distinct valid regex queries `Q1` through `Q4`.

The fixed probe is `Q1`, executed with the released deterministic `grep` tool
using `before=2`, `after=8`, and `max_matches=30`.

Phase two, five concurrent calls:

- model: `google/gemini-2.5-flash`;
- non-thinking;
- seed `38000`;
- temperature `0`;
- maximum output `2,000` tokens.

Each response must contain:

- one integer likelihood in `[0,100]` for each `H1` through `H6`, interpreted
  as a relative score for `P(probe observation | hypothesis, Q1)`;
- exactly three follow-up actions `F1` through `F3`.

A follow-up action has only `tool` and `argument`:

- `grep`: `argument` is a valid regex;
- `view_log_lines`: `argument` is a positive base-10 line number and the
  runtime uses radius `30`.

At least one follow-up should use a distinctive literal or line number learned
from the probe observation and absent from both the initial context and prior
tool arguments.

Expected requests: exactly `10`. Concurrency: `5`. No retry, repair,
continuation, reissue, response normalization, model substitution, or reasoning
fallback. Projected cost: `$0.12`; hard run cap: `$0.30`.

## Frozen Gates

All gates are conjunctive:

- exactly `10` accepted requests and `10` HTTP attempts;
- zero retries, provider-error retries, reasoning tokens, and forced exits;
- all five planner and updater responses parse exactly;
- every case has six unique hypotheses;
- every case has four unique, non-generic, compilable regex queries;
- `Q1` returns one or more log matches on at least four of five cases;
- every updater has all six likelihoods;
- at least four of five cases have likelihood range at least `20` points and
  at least three distinct likelihood values;
- all `15` follow-up actions parse and execute without a tool error;
- at least four of five cases contain a corrected observation-dependent
  follow-up;
- at least two corrected dependency types occur across cases;
- total cost is at most `$0.30`;
- no ground truth, diagnosis evaluator, policy endpoint, or confirmation case
  is accessed.

Failure closes this exact interface. There is no response repair or case
removal. A transport-only schema rejection may permit one prospectively frozen
codec simplification, but semantic gate failure does not.

Passage authorizes only a separately preregistered, small first-link ranking
experiment on development cases. It does not authorize a powered policy claim.

## Dry Verification

Before any real response:

- focused parser, dependency, source-boundary, and gate tests must pass;
- an exact ten-call deterministic fixture must pass;
- implementation, tests, and this preregistration must be committed and
  pushed.
