# Zendo Structured-Output Serving Smoke Preregistration

Date frozen: 2026-07-25, before any structured-output response.

## Purpose

The seven-rule free-form-AST confirmation stopped before scientific execution
when GPT-5.4 paired `attribute="color"` with `value="large"`. This is a
cross-field schema failure. OpenRouter documents strict JSON Schema structured
outputs for GPT-5.4, including provider routing that requires schema support.

This is a serving-only gate on three already-open development rules: `zeta`,
`phi`, and `mu`. It does not use any fresh confirmation rule, generate any branch
tree, call a scorer, or evaluate hidden truth. A pass only establishes that the
provider can return executable recursive rule ASTs under the strict schema.

## Frozen protocol

- Interface: `zendo-structured-output-serving-1`.
- Model: `openai/gpt-5.4`, temperature zero, explicit non-reasoning.
- Tasks: `zeta`, `phi`, `mu`.
- Exactly three physical requests and three HTTP attempts.
- OpenRouter `response_format.type="json_schema"`, `strict=true`.
- Provider routing sets `require_parameters=true`.
- The recursive schema couples each attribute to its legal value type:
  - color: blue/red/green;
  - size: small/medium/large;
  - orientation: upright/left/right/strange;
  - grounded: boolean.
- It also constrains all AST variants, required fields, no extra properties,
  counts 0--6, exactly 12 particles, and IDs `H01`--`H12`.
- Existing executable parser remains the final validator.
- No response healing, normalization, repair, replacement, or retry.
- Projected cost `$0.08`; hard cap `$0.15`.
- OpenRouter only; no OatML; preserve the `$25` reserve.

## Frozen gates

All must pass:

- exactly three adapter requests and HTTP attempts;
- zero transport retries, reasoning tokens, and forced exits;
- all three responses satisfy both provider schema and the executable parser;
- each 12-particle population has at least eight unique ASTs;
- total cost at most `$0.15`.

Failure closes this schema implementation. Pass authorizes a separately committed
structured-output amendment to the seven-rule confirmation, with fresh responses
and unchanged scientific tasks, roots, controls, endpoints, and thresholds.
