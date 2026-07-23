# Range-Gated Rock Successor-Grounded 26B Smoke Result

Status: **failed closed before policy scoring; conditional S1 not run.**

The first seed `24179` cell exhausted both bounded responses. Gemma 4 26B A4B used
`6,505` reasoning tokens across the two requests, hit forced finalization twice, and
returned the literal final text `None` both times. Neither response was valid JSON, so
zero cells were accepted and no route or plan value was scored.

This is a serving-budget failure, not evidence about successor grounding or
horizon-three proposal quality. The preregistered line required zero forced exits and
forbade a budget or prompt change, so it stops without S1.

## Usage

- Model: `google/gemma-4-26b-a4b-it`, thinking enabled
- Requests: `2`
- Prompt tokens: `3,253`
- Completion tokens: `8,704`
- Reasoning tokens: `6,505`
- Forced exits: `2`
- Cost: `$0.00331264`
- Project spend after run: `$38.51874830 / $110`

The exact failure texts and accounting are retained in
`range_gated_rock_successor_grounded_26b_smoke_20260723/SMOKE_FAILURE.json`.
