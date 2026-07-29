# CUPID Active-Preference Serving Smoke V1 Failure

Status: **failed closed before hidden-target access; V1 interface closed**.

Date: 2026-07-29

Public failure:
`results/nonmyopic/cupid_active_preference_serving_smoke/cupid-active-preference-serving-20260729T003945Z/SERVING_FAILURE.json`.
SHA-256:
`a65e45184c26e1380e113fbc99088ad99741cbc1cb727e5c7cdfa2a4f5ed7e71`.

Private raw-response SHA-256:
`c3552a4339dfff3c7bdc1f405f527f7662b2475c4e0cec4e8555ef003793fd6d`.

## Execution

The planner phase made exactly five concurrent `openai/gpt-5.4-mini`
requests. All five completed on their first HTTP attempt:

- accepted requests / HTTP attempts: `5 / 5`;
- retries and provider-error retries: `0 / 0`;
- reasoning tokens and forced exits: `0 / 0`;
- prompt / completion tokens: `15,955 / 3,203`; and
- reported cost: `$0.02637975`.

The run stopped while parsing the saved planner responses. No
`google/gemini-2.5-flash` target request was made. Consequently no released
hidden preference, checklist, development row, holdout row, or policy endpoint
was accessed.

## Exact Failure

The structured-output schema constrained each `answer_signature` to a string
of exactly six characters, while the prompt requested six `0/1` bits. Two of
five planner responses used the intended compact representation for all twelve
hypotheses. Three responses instead inserted spaces or commas between bits.
The provider honored the six-character schema length, leaving values such as
`"1 0 1 "` and `"1,1,0,"`, which encode only three complete bits and fail the
frozen `[01]{6}` parser.

This is a representation/serving failure. The frozen support-diversity,
partition-entropy, independent-target-coverage, and target-signature gates were
not evaluated.

## Decision

V1 remains failed and will not be repaired, retried, resumed, or rerun on its
five serving cases. Its aggregate result is immutable.

Because the failure occurred before any hidden-target access and says nothing
about preference-support mechanics, one disjoint V2 serving qualification is
permitted. V2 must be frozen before responses, use five previously unopened
development rows, keep the same models, prompts, two-phase control flow, gate
thresholds, and cost cap, and change only the signature JSON representation
from a six-character string to an array of six schema-enforced `0/1` integers.
V2 is a new interface, not a relabeling of V1.
