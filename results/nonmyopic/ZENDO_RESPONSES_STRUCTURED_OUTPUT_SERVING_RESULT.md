# Zendo Responses Structured-Output Serving Result

Date: 2026-07-25

Status: **routing failed before generation; zero cost**.

Run: `zendo-responses-structured-serving-20260725T074656Z`

All three `/api/v1/responses` attempts returned OpenRouter HTTP 404: no endpoint
for GPT-5.4 accepted the required structured text parameters with
`require_parameters=true`.

- HTTP attempts: `3`
- Completed responses: `0`
- Tokens / cost: `0 / $0`
- Branches, scorers, or endpoints: `0`

No schema fallback or response healing was used. Together with the prior Chat
Completions routing null, this closes provider-enforced recursive JSON Schema for
this experiment.

- Public failure:
  `results/nonmyopic/zendo_responses_structured_output_serving/RESULT.json`
- Failure artifact SHA-256:
  `fe1460dbfbe9601995fafa16e1ee748238973baed318476d09f2911b2da0f9ec`
- Empty raw checkpoint SHA-256:
  `e5b5f348c21d8cbfc4c7b8ddce20d067aadffa22c0b6bc0d14339c77818fafa9`
