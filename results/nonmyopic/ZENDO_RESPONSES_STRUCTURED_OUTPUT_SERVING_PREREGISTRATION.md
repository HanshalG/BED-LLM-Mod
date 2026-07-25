# Zendo Responses Structured-Output Serving Smoke Preregistration

Date frozen: 2026-07-25, before any Responses API request.

## Purpose

The Chat Completions structured-output smoke made zero model calls because
OpenRouter could not route GPT-5.4 with required schema parameters on that
endpoint. OpenRouter separately documents its OpenAI-compatible Responses API at
`/api/v1/responses`, where structured text output is configured through
`text.format`.

This is a distinct serving-only gate on the same already-open `zeta`, `phi`, and
`mu` rules. It does not touch any confirmation endpoint.

## Frozen protocol

- Interface: `zendo-responses-structured-output-serving-1`.
- Endpoint: `POST https://openrouter.ai/api/v1/responses`.
- Model: `openai/gpt-5.4`, temperature zero, reasoning effort `none`.
- Input uses stateless structured message arrays.
- Output uses `text.format.type="json_schema"` with the already committed strict
  recursive Zendo schema.
- Provider routing requires all parameters.
- Exactly three HTTP attempts and, if routed, three completed responses.
- Existing executable parser validates every returned population.
- No response healing, fallback, repair, normalization, or retry.
- No branches, scorers, hidden truth, or scientific endpoints.
- Projected cost `$0.08`; hard cap `$0.15`; OpenRouter only; no OatML.

## Frozen gates

All must pass:

- exactly three completed adapter responses and HTTP attempts;
- zero retries, reasoning tokens, and forced/incomplete responses;
- all three responses satisfy the schema and executable parser;
- each population has at least eight unique ASTs;
- cost at most `$0.15`.

Failure closes the Responses API schema path. Pass authorizes only a separately
committed structured seven-rule confirmation amendment with fresh responses and
unchanged scientific tasks, controls, endpoints, and thresholds.
