# Range-Gated Rock Successor-Grounded Qwen 14B Smoke Result

Status: **failed closed before policy scoring; conditional S1 not run.**

The generic endpoint-free serving calibration had returned exact JSON after 155
reasoning tokens. Under the full successor-grounded environment prompt, however, both
bounded seed `24181` responses consumed the complete 4,352-token output allowance,
hit forced finalization, and returned the literal final text `None`.

Zero cells were accepted, so no route or plan value was scored. The registered line
required zero forced exits and forbade a reasoning-budget change; it therefore stops
without S1. Together with the Gemma OpenRouter failure, this shows that the current
single-call OpenRouter adapter does not reliably obtain a final channel when these
open-weight models exhaust extended reasoning. It does not test horizon-three policy
quality.

## Usage

- Model: `qwen/qwen3-14b`, thinking enabled
- Requests: `2`
- Prompt tokens: `2,945`
- Completion tokens: `8,704`
- Reasoning tokens: `8,329`
- Forced exits: `2`
- Cost: `$0.00238346`
- Project spend after run: `$38.52117794 / $110`

The retained failure is in
`range_gated_rock_successor_grounded_qwen14b_smoke_20260723/SMOKE_FAILURE.json`.
