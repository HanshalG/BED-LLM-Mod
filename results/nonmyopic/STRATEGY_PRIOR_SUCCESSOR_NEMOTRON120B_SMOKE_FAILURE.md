# Nemotron 3 Super 120B Successor Serving-Gate Failure

The registered Nemotron smoke used `reasoning.max_tokens: 768` with total
`max_tokens: 1536`. Its first strict L1 cell returned no final content, hit the total
completion limit, and was therefore rejected as invalid JSON. OpenRouter reported
1,536 completion tokens, 1,541 reasoning tokens, and one forced exit, so this is a
serving-budget failure rather than an evaluation of strategy quality.

The failed-closed artifact is
`strategy_successor_nemotron120b_interface_smoke/20260718/SMOKE_FAILURE.json`.
It cost `$0.00162960` for one request and did not launch a policy endpoint.

The registered medium-reasoning/4,096-token amendment is justified by this specific
budget exhaustion and leaves the frozen 30-trial endpoint, controls, seed, and pass
rule unchanged. It must pass the same ten first-response, no-repair, zero-forced-exit
gate before any endpoint run is permitted.
