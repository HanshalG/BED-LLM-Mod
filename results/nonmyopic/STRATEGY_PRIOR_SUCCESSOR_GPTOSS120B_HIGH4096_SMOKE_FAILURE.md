# gpt-oss-120B High/4096 Serving-Gate Failure

The initial gpt-oss smoke used mandatory high reasoning with `max_tokens: 4096`. It
failed on the first L1 cell before an endpoint: the final content was `None`, all 4,096
completion tokens were reasoning tokens, and the response ended by length. The request
cost `$0.00072555`.

This does not evaluate gpt-oss plan quality. Official OpenRouter reasoning guidance
states that an effort request allocates a fraction of `max_tokens` to reasoning and
that `max_tokens` must leave room after that allocation for a final response. The
high/4096 configuration did not reliably do so for this prompt. A separate corrected
configuration is registered before another smoke: medium reasoning with 8,192 total
tokens, preserving roughly 4k tokens for both reasoning and final JSON.

Machine-readable artifact:
`strategy_successor_gptoss120b_interface_smoke/20260718/SMOKE_FAILURE.json`.
