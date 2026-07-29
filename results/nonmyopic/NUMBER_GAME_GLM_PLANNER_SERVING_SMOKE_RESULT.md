# Number Game GLM Planner Exact-10 Serving Result

Date completed: 2026-07-29.

Status: **failed closed at serving; no formal run authorized**.

## Result

GLM 5.1 completed 10 responses, each with `stop` finish reason. Nine responses
were exact JSON objects. One response wrapped its otherwise nonempty content
in a Markdown code fence, so the frozen exact-JSON parser stopped before any
support or efficacy aggregate was formed.

The run used 2,563 prompt tokens and 7,309 completion tokens and cost
`$0.030343251`. The provider reported three reasoning tokens across three
responses despite explicit nonreasoning. Thus both exact transport and
zero-reasoning gates fail independently.

## Decision

No fence stripping, normalization, response replacement, reissue, or partial
support scoring is performed. The GLM formal run is not launched. Per the
preregistration, this failure ends further planner-family screening under
this interface.

Artifacts:

- public `FAILURE.json` SHA-256:
  `cd7e1465a7de5ac7883cbccb2e063b70d78cc43519ee6d019df30d35e77bcc13`;
- private raw-response SHA-256:
  `de5ccac62bf1575c0c30bab92049589467b6d3200738cc9ca2b5a3db195f7de1`.
