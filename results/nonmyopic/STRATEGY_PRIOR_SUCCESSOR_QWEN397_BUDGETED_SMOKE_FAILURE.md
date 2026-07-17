# Qwen 397B Explicit-Budget Serving-Gate Failure

The explicit-Qwen amendment sent `reasoning.max_tokens: 512` with `max_tokens: 768`.
It still failed closed on the first L1 strategy cell before an endpoint: the provider
reported a length finish, 768 completion tokens, and final content `None`. The request
cost `$0.00266445`.

The reported reasoning-token count was one, so the provider did not honor the intended
Qwen reasoning/final allocation in a useful way for this route. This is a serving
failure. The successor therefore switches to a model whose live catalog explicitly
advertises fixed reasoning-token-budget support.

Machine-readable artifact:
`strategy_successor_qwen397_budgeted_interface_smoke/20260718/SMOKE_FAILURE.json`.
