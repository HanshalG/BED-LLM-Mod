# DeepSeek V4 Pro Successor Serving-Gate Failure

The registered DeepSeek swap used `deepseek/deepseek-v4-pro` with provider-native
medium reasoning and `max_tokens: 2048`. It failed closed on the first no-repair Rock
L1 cell, before any policy endpoint was evaluated.

The raw returned final content was `None`. The provider reported 2,048 completion
tokens, all tagged as reasoning tokens, and a `length` finish reason. The single smoke
request cost `$0.00361018476`. This is another serving-budget failure, not evidence
about the StrategyEIG policy. The successor registration therefore swaps the model
rather than rerunning DeepSeek.

Machine-readable artifact:
`strategy_successor_deepseek_v4_pro_interface_smoke/20260718/SMOKE_FAILURE.json`.
