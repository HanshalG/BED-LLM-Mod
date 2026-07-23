# Range-Gated Rock Depth-Four GPT-5.4 Smoke Result

Status: **failed closed before cell acceptance; conditional S1 was not run and
the frontier h4 line stops.**

Full GPT-5.4 high reasoning had passed the endpoint-free three-route calibration
with compact final answers. Under the full h4 prompt, all ten fresh seed-24224
cells exhausted the 4,352-token output allowance on both the initial response
and registered correction. Every response contained reasoning usage but empty
final content.

The current OpenRouter adapter invokes its bounded reasoning-disabled
finalization only for model specs marked `thinking: true`. This GPT spec used
the native `reasoning_effort: high` field, so none of the 20 length stops
triggered a finalization call. Zero cells entered the provider and no route,
plan value, or scientific endpoint was scored.

## Usage

- accepted cells: `0/10`
- invalid empty responses: `20`
- physical requests: `20`
- length stops: `20`
- bounded forced-final requests: `0`
- prompt tokens: `139,858`
- completion/reasoning tokens: `87,040 / 87,040`
- cost: `$1.522765`
- project spend after S0: `$40.23943364 / $110`

This is a serving-budget failure, not evidence that GPT-5.4 could not identify
the h4 route. The preregistration nevertheless forbids a model, prompt, adapter,
budget, projection, or seed repair after S0 failure. Seed-24225 S1 is
permanently unauthorized and no further h4 model line is launched.

The banked h4 evidence therefore remains:

1. a strong independently audited exact d4-over-d3 structural effect; and
2. a measured transfer boundary in which three open-weight models failed
   reliable score-free proposal serving, while the one frontier attempt did not
   reach a final channel.

Artifacts:

- `range_gated_rock_depth4_gpt54_smoke_20260723/SMOKE_FAILURE.json`
- `range_gated_rock_depth4_gpt54_smoke_20260723/run.log`
