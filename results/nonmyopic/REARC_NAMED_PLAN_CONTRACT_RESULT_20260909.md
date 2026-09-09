# Named-plan contract parity verified

Previous goal turn exposed an engineering failure, not a scientific negative.
This turn replaces the successor's array-of-ID plan with four mandatory named
string fields, p0..p3. The exact API schema is used by a standard local JSON Schema
validator. MaxLength512 means characters in both places; complete request bytes
still have a separate64KiBcap. Field order is irrelevant and text is not truncated,
normalized or silently rewritten. Old implementation and response remain closed.

Named compile/repair update and request contract are connected and tested in both
contrasting and ordinary modes. Both retain3calls and16compiled attempt slots.
Whitespace-only plans are structurally valid and explicitly reported as semantic
diagnostics. Program syntax/runtime validity is separate from response schema.
None of these structural tests is evidence of correct hypotheses or useful diversity.

49tests pass in0.41s: ASCII, multibyte, non-BMP and combining-character strings at
0/1/511/512/513characters; escaped/raw JSON representations; all24field orders;
missing/extra/non-string fields; duplicate JSON keys; schema-copy isolation;
Unicode plan-to-compilation transport; executable-invalid/schema-valid outputs;
full3-call updates; and frozen old discrepancy/interface regressions. The old
one-call failure still replays exactly with no new call.

Dependency environment is isolated at /private/tmp/bed-plan-contract-env using
the existing Anaconda Python base; requirements-plan-contract.txt pins the new
validator dependencies. The ordinary Anaconda interpreter lacks jsonschema;
use the isolated interpreter (or an equivalently pinned environment), not the
old Python invocation, for this successor. No primary environment was modified.

REARC_NAMED_PLAN_SUPPORT_PROTOCOL_20260909.md freezes identical scientific gates
and budget with a fresh metadata-only six-task cohort excluding all26prior IDs.
No new cohort has been selected, generated or queried this turn. Next is wiring
the named interface through controller/source/budget/replay and verifying source
and complete failure paths before another paid attempt. No paid authorization yet.

Cost0, account balance23.482712431/conservative London Sep9 spend1.09900909
unchanged. This corrects a preventable blocker but proves no non-myopic improvement.
The research objective remains unachieved.
