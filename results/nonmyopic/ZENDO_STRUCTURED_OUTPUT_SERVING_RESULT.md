# Zendo Structured-Output Serving Smoke Result

Date: 2026-07-25

Status: **failed before generation; zero cost**.

Run: `zendo-structured-output-serving-20260725T074350Z`

OpenRouter rejected all three concurrent Chat Completions requests with HTTP 404:
no endpoint for `openai/gpt-5.4` could handle the required strict JSON Schema
parameters while `provider.require_parameters=true`.

No model response, token, adapter request, or charge was recorded. There was no
fallback that silently dropped the schema requirement.

- HTTP attempts: `3`
- Adapter/model responses: `0`
- Tokens: `0`
- Cost: `$0`
- Scientific endpoints: `0`

This closes the Chat Completions structured-output path. It does not test the
schema itself or the scientific method. OpenRouter separately documents
structured text formats on its Responses API; that would be a distinct serving
interface and must receive its own preregistered smoke.

- Public failure:
  `results/nonmyopic/zendo_structured_output_serving/RESULT.json`
- Failure artifact SHA-256:
  `7c94d457fa667869227ef0118589465a7789a7bdb3cc9e068a21b4e8eecc794c`
- Empty raw checkpoint SHA-256:
  `ce3efc89a0bad9ce93d4f78021ef4713ba4422ff4cc9516f07c50eb0004e5200`
