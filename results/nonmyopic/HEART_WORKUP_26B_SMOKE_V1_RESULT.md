# Cleveland Heart Workup 26B Serving Smoke V1

The preregistered S0 interface **failed closed** before any proposal-quality endpoint.
Job `106372` initialized non-thinking Gemma 4 26B successfully on `oat14`, accepted
three cells, and then exhausted the one-retry allowance on cell three.

Both invalid outputs had the correct four arrays and branch counts, but copied indices
`9` and `10` from the 12-item workup-root menu into ordinary-query roots whose local
menus permitted only `0..3`. The repair attempt fixed one root but repeated the same
scope error in two others. This revision therefore stops as registered.

Usage was five physical requests, 7,549 prompt tokens, 192 completion tokens, zero
reasoning tokens, zero forced exits, zero rollout/scoring calls, and `$0` API cost.
Jobs `106369`--`106371` were infrastructure-only failed starts before model
initialization and made no requests.

The failure supports one separately registered prompt-only repair: append explicit
per-root local index limits at the end of the prompt and state that indices cannot be
copied between roots. No policy score was computed or inspected, and the frozen S1
cells, controls, thresholds, and seed remain unchanged.

Artifact:
`results/nonmyopic/heart_workup_26b_smoke_20260723/SMOKE_FAILURE.json`.
