# COPEx Direct-Proposal DeepSeek Quality-Screen Amendment

Recorded on 2026-07-18 before any DeepSeek direct-proposal request.

The Gemma 26B thinking route is rejected at interface level. Its original 2,048/512
cell and its 4,096/1,024 replacement each used the full allocation on both permitted
attempts, returned final content `None`, and failed before yielding one valid angle
cell or any proposal-quality outcome. The corresponding raw artifacts are under
`results/nonmyopic/copex_direct_proposals_thinking_probe/20260718/` and `20260718b/`.

The same fixed eight-state quality screen now substitutes
`deepseek/deepseek-v4-flash` with provider-native `reasoning_effort: high` and an
8,192-token completion allowance. This endpoint is already serving-verified in this
repository: ten strict continuous/rock strategy JSON cells completed with zero forced
exits and zero repair attempts. The direct-angle prompt, state set, parser, exact
quadrature pool scoring, non-thinking/grid comparators, and promotion rule from the
thinking-quality preregistration are unchanged.

This remains a mechanism screen. A positive screen authorizes only a fresh paired
DeepSeek d2 pilot; it does not create a policy result itself.
