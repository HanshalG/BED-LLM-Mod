# AgentClinic MedQA dynamic-support source result

Date: 2026-08-13

Status: **source failed closed; exact construction closed**

The source protocol was frozen before any MedQA field value or case identity was
opened. It bound the immutable extended MedQA file to exactly 213 nonblank rows
and one exact five-field `OSCE_Examination` schema.

The value-blind audit found:

- immutable repository, code, and data hashes: pass;
- observed nonblank rows: `214`, not `213`;
- exact top-level schema: `214/214`;
- distinct OSCE schemas: `2`, not `1`;
- exact registered five-field OSCE schema: `213/214`.

The count mismatch arose because the final JSONL record has no trailing newline,
so line count was not record count. Independently, one record contains a sixth
OSCE field, so the exact nested-schema gate also fails under the actual 214-row
population.

The audit stopped at population shape. It did not serialize any case ID, source
value, objective, patient fact, finding, test result, or diagnosis. Intake
separation, leak checks, mechanics cases, model responses, likelihoods, planning
scores, and diagnosis endpoints remained unopened.

No 213-row subset, corrected 214-row protocol, or original-MedQA fallback is
authorized after observing this mismatch. This result says nothing about policy
efficacy; it is a zero-call source-integrity null.

- OpenRouter calls/cost: `0` / `$0`
- OATML cluster use: none
