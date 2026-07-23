# Range-Gated Rock Successor-Grounded Cluster 26B Smoke Result

Status: **failed preregistered S0 serving gate; conditional S1 not run.**

Direct vLLM repaired the missing-final problem. Seed `24183` accepted six consecutive
cells on their first response, and all six contained the exact delayed on-site plan
`move-SOUTH, move-SOUTH, check-5`.

Cell 6 then failed both bounded attempts. Each raw response began with a syntactically
complete four-key JSON object whose south-root plan was again the exact delayed route,
but Gemma appended self-correction prose after the closing brace. The frozen parser
required the entire response to be JSON, so both objects were rejected and the
ten-cell gate failed closed.

No plan value was inspected before the terminal failure, and S1 was not launched.
This is now a parser-only serving failure rather than a spatial or horizon-credit
failure: the route appeared in all `6/6` accepted cells and both rejected JSON
prefixes. A fresh line may preregister strict first-object extraction, but this seed
cannot be reused.

The vLLM usage snapshot records 16 physical generations (eight first-stage reasoning
generations and eight bounded final generations), `59,664` prompt tokens, `34,068`
completion tokens, and `$0` API cost. The provider retained six accepted logical
cells and two invalid responses. Full first-stage reasoning was not written because
this standalone config had no `log_path`; the raw final-channel text and all compiled
accepted policies are retained in
`range_gated_rock_successor_grounded_cluster26b_smoke_20260723/SMOKE_FAILURE.json`.

Cluster job: `106384`, `msc`, node `oat14`.
