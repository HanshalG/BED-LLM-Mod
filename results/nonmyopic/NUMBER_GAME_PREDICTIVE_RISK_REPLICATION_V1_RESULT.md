# Number Game Predictive-Risk Replication V1 Result

Date: 2026-07-28

Protocol:
`results/nonmyopic/NUMBER_GAME_PREDICTIVE_RISK_REPLICATION_PREREGISTRATION.md`

Run:
`results/nonmyopic/number_game_predictive_risk_replication/number-game-predictive-risk-replication-20260728T130000Z`

## Verdict

**Transport failure; replication science unmeasured.**

V1 completed one of eight trees. During the second tree, one Gemini branch
request returned an HTTP-success response with `finish_reason="error"`, zero
completion tokens, and zero cost. The returned content was not valid JSON. The
other 34 responses completed with `finish_reason="stop"`. The runner failed
closed before computing or exposing aggregate scientific metrics.

This was not a token-cap failure: normal branch responses used 804--996
completion tokens against a 4,200-token limit. It was also not a schema,
predicate, or scientific-gate failure. The adapter did not classify provider
completion errors as retryable transport failures.

V1 is not resumed or partially scored. A transport-only V2 may:

- retry only responses explicitly marked `finish_reason="error"` when their
  reported cost is zero;
- preserve the exact scientific method, controls, thresholds, temperature,
  tree count, and whole-tree aggregation; and
- use eight entirely fresh planning and target seeds so the completed V1 tree
  is not selected into the replication.

## Accounting

- HTTP responses before failure: `35`
- Normal stops / provider errors: `34 / 1`
- Reasoning tokens / length exits: `0 / 0`
- Logged cost: `$0.0837777`
- Completed trees: `1 / 8`
- Private partial-response SHA-256:
  `636249d896e59e7d11046d57787ab68ee361bf26249826ed2dd88fd8c564add1`
