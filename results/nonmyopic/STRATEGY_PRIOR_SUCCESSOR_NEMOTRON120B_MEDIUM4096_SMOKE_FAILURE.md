# Nemotron 3 Super 120B Medium/4096 Serving-Gate Failure

The registered medium-reasoning amendment was executed exactly once. Its first strict
L1 cell again returned no final content and was rejected as invalid JSON. OpenRouter
reported 4,096 completion tokens, 3,892 reasoning tokens, and one forced exit.

This confirms that the current OpenRouter Nemotron route does not reserve answer tokens
under either its explicit reasoning-budget or medium-effort setting. The result is
strictly a serving-interface failure: one request, `$0.00393360`, and no policy
endpoint. The separate successor registration remains open to a different
serving-verified model.

Artifact:
`strategy_successor_nemotron120b_medium4096_interface_smoke/20260718/SMOKE_FAILURE.json`.
