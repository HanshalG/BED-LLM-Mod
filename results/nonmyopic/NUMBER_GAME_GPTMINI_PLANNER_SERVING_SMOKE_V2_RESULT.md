# Number Game GPT-5.4 Mini Planner Serving Smoke V2 Result

Run: `number-game-gptmini-planner-serving-v2-20260729T062946Z`

Status: **gated null**. The single-draw GPT-Mini interface remains closed.

## Result

- Exactly 10/10 accepted calls and HTTP attempts.
- Zero retries, provider-error retries, reasoning tokens, or forced exits.
- Cost: `$0.026847`.
- Initial generated supports: 23 and 23 valid unique hypotheses.
- Conditioned generated supports: 8, 8, 14, 8, 8, 12, 13, and 1.
- Merged first supports: 10, 25, 22, and 22.
- Merged second supports: 9, 17, 19, and 12.

All retained-rejuvenation support-size gates passed. The frozen requirement
that every conditioned LLM draw contribute at least four valid hypotheses
failed because `42=NO,75=NO` contributed only one. The stronger literal
substitution prompt therefore did not make single-draw semantic consistency
reliable enough for a new 1,856-call cohort.

Public result SHA-256:
`327bd2e643447148bb91543733e0101c5bfa3c879f3668bec2e822e4671d9eb4`.

Private raw-response SHA-256:
`55836bf5f58538fadd9feba278ace9da84f58684adb690aa80ee952db357836b`.

No efficacy endpoint was evaluated. This closes further prompt-only
GPT-Mini single-draw serving attempts. A future GPT-Mini experiment would
need a method change such as pooled independent belief draws, not another
wording revision.
