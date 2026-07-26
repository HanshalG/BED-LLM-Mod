# KnowU Dynamic-Support Serving V2 Result

Date: 2026-07-26

Status: **failed closed before mechanics**

Prompt-only flat JSON reached GPT-5.4 successfully, but the first synthetic
batch produced at least one clarification question containing a conjunction.
The frozen atomic-question parser rejected it. Because raw responses were
checkpointed only after parsing in V2, the malformed response itself was not
retained; V3 must checkpoint every response batch before parsing.

Recorded usage:

- HTTP attempts and accepted requests: 4
- prompt tokens: 1,476
- completion tokens: 1,082
- reasoning tokens: 0
- retries and forced exits: 0
- cost: $0.019920

No mechanics request or scientific endpoint ran. The public failure artifact
is preserved at
`knowu_dynamic_support_serving/knowu-dynamic-support-serving-v2-20260726T012647Z/GATE_FAILURE.json`.

V3 may only:

- make the existing atomicity instruction explicit by forbidding the literal
  words `and` and `or` in generated questions;
- checkpoint raw batches before parsing;
- remove the accidental internal `reasoning_effort="none"` string so the
  adapter sends no reasoning request at all.

It may not relax the atomicity parser or alter any scientific criterion.
