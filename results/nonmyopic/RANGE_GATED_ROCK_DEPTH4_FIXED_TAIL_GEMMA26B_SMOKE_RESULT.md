# Range-Gated Rock Depth-Four Gemma 26B Smoke Result

Status: **failed the frozen route and exact-selection gates; conditional S1 was
not run.**

OpenRouter Gemma 4 26B A4B thinking completed all ten fresh seed-24208 cells with
four dynamically legal machine-rooted plans. The failure was proposal quality,
not parsing or exact scoring:

| Frozen S0 endpoint | Result | Required |
| --- | ---: | ---: |
| Critical `N,N,N,check-4` route present | `0/10` | `>=8/10` |
| Exact h4 root selected | `0/10` | `>=8/10` |
| Accepted legal cells | `10/10` | `10/10` |
| Scoring-time model calls | `0` | `0` |

The characteristic error was a one-step travel miscount. For the north root,
Gemma commonly proposed `move-NORTH, move-WEST, move-WEST, check-6`. This ends at
`(4,5)`, while rock 6 is at `(3,5)`, so the check remains weak and remote. Other
plans similarly spent an initial check and then acted as if only two moves were
needed to reach rock 4. The prompt exposed the full score-free transition graph,
rock coordinates, and three tail slots; all accepted plans were legal, but none
composed the registered three-move route to an on-site assay.

Two first responses were invalid and succeeded on the frozen correction attempt.
The serving adapter handled reasoning-only length stops correctly:

- accepted logical cells: `10`
- invalid first responses: `2`
- physical OpenRouter requests: `23`
- forced exits: `12`
- forced-final requests/successes: `11/11`
- prompt tokens: `175,575`
- completion tokens: `52,977`
- reasoning tokens: `41,829`
- cost: `$0.03729987`
- project spend after S0: `$38.56015510 / $110`

Because both registered route gates failed, seed-24209 S1 is permanently
unauthorized for this Gemma line. This result is evidence of a multi-step spatial
composition limit in the 4B-active MoE model, not evidence against the exact h4
planning effect.

Artifacts:

- `range_gated_rock_depth4_fixed_tail_openrouter_smoke_20260723/SMOKE.json`
- `range_gated_rock_depth4_fixed_tail_openrouter_smoke_20260723/run.log`
