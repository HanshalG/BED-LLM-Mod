# Range-Gated Rock Depth-Four Qwen Projected Smoke Result

Status: **failed the frozen route and contribution gates; conditional S1 was
not run.**

The fresh seed-24216 bounded-projection smoke completed all ten logical cells and
compiled four legal plans per cell. It did not recover the load-bearing h4
route:

| Frozen S0 endpoint | Result | Required |
| --- | ---: | ---: |
| Critical `N,N,N,check-4` route present | `0/10` | `>=8/10` |
| Exact h4 root selected | `0/10` | `>=8/10` |
| Selected plans fully LLM-authored | `6/10` | `10/10` |
| Cells retaining at least one LLM branch | `8/10` | `10/10` |
| Projected branch fraction | `15/40` | `<=30/40` |
| Legal compiled cells | `10/10` | `10/10` |

Projection behaved as registered: every projected branch was its fixed root
followed by three `check-0` actions, and no projection created the critical
route. The quality failure therefore remains attributable to Qwen. Typical
north-root plans again stopped short of an on-site assay, such as
`N,W,W,check-6`, or spent a tail action undoing movement. In two cells every
model branch remained invalid after correction and all four branches were
projected. The exact scorer selected a projected check plan in four cells,
independently failing the LLM-contribution requirement.

The exact critical route that appeared once in the prior closed seed-24212
correction response did not replicate on any of these ten fresh cells. This
rules out treating that single failed-cell route as evidence of reliable dense
Qwen h4 planning.

## Serving

- logical cells: `10`
- invalid attempt records: `18`
- physical OpenRouter requests: `32`
- forced exits: `16`
- forced-final requests/successes: `12/8`
- prompt tokens: `223,293`
- completion tokens: `87,444`
- reasoning tokens: `79,192`
- cost: `$0.05902517`
- project spend after S0: `$38.62723769 / $110`

Seed-24217 S1 is permanently unauthorized. The exact h4 structural result
remains valid, but neither 4B-active Gemma nor dense Qwen 14B reliably supplies
the required spatial continuation under this score-free interface.

Artifacts:

- `range_gated_rock_depth4_qwen14b_projected_smoke_20260723/SMOKE.json`
- `range_gated_rock_depth4_qwen14b_projected_smoke_20260723/run.log`
