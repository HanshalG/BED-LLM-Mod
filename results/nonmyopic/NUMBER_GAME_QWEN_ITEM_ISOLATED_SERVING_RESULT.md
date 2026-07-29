# Number Game Qwen Item-Isolated Serving Result

Run: `number-game-qwen-item-isolated-serving-smoke-20260729T093300Z`

Status: **serving gate failed; the conditional 32-tree cohort is not run**.

## Result

All ten Qwen responses were accepted, valid strict JSON documents. There were
zero HTTP or provider retries, zero reasoning tokens, zero forced exits, and
the run cost `$0.01149504`.

The parser and deployed retained-support mechanics were healthy:

- initial valid counts: `23, 23`;
- merged first-support counts: `21, 24, 17, 26`;
- merged second-support counts: `7, 4, 24, 26`;
- no response required complete-item salvage.

The frozen per-draw generation gate failed. Conditioned generated-support
counts were `17, 19, 15, 16, 6, 2, 22, 16`; the two-valid draw is below the
required minimum of four. It had 18 duplicate extensions, two inconsistent
rules, and two invalid expressions.

## Consequence

The response with only two valid new extensions still merged with two valid
retained parent extensions, so every deployed merged-support minimum passed.
That is a useful diagnostic, but the raw-child threshold was fixed before the
response. It is not relaxed after the fact.

The item-isolated single-draw Qwen route is closed and the registered fresh
32-tree cohort is not authorized. A future pooled-independent-draw method
would be a new variance-reduction method with new seeds and gates, not a rerun
or reclassification of this smoke.

Public `RESULT.json` SHA-256:
`984086f40896e648cbf84f1c82a25de8c7aaee8b83e42d69d85b694fba200f82`.

Private raw-response SHA-256:
`8cba3a23d3e07926820dfb36bc96c4751daf012524d8cc792134c425c13740c4`.
