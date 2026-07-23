# Range-Gated Rock Depth-Four Qwen 32B Smoke Result

Status: **failed closed; conditional S1 was not run and the registered
open-weight model search stops.**

The fresh seed-24220 dense Qwen 3 32B smoke ran ten cells concurrently. Seven
cells returned complete legal plan sets; three exhausted the first response and
registered correction without an acceptable full object. Because the frozen
gate required all ten cells before scoring, no S0 route-rate or policy endpoint
was computed.

Among the seven retained legal cells, the critical
`move-NORTH, move-NORTH, move-NORTH, check-4` plan appeared in `3/7`.
Other north-root plans commonly stopped short at a remote check. One rejected
cell did identify the critical route content but returned it as four tail
actions, repeating the already-fixed north root; its correction then truncated
inside a JSON fence. This is useful diagnostic evidence but cannot enter the
registered endpoint.

## Serving

- accepted legal cells: `7/10`
- invalid attempt records: `10`
- physical OpenRouter requests: `27`
- forced exits: `19`
- forced-final requests/successes: `10/2`
- prompt tokens: `187,441`
- completion tokens: `134,807`
- reasoning tokens: `125,591`
- cost: `$0.07910875`
- project spend after S0: `$38.70730114 / $110`

Seed-24221 S1 is permanently unauthorized. Per the preregistration, this ends
further open-weight score-free h4 model substitution. The evidence now supports
a clear capacity boundary: the exact h4 effect is strong, but Gemma 4B-active,
Qwen 14B dense, and Qwen 32B dense do not reliably serialize and compose the
required four-step proposal set under the same no-utility-answer interface.

Artifacts:

- `range_gated_rock_depth4_qwen32b_smoke_20260723/SMOKE_FAILURE.json`
- `range_gated_rock_depth4_qwen32b_smoke_20260723/run.log`
