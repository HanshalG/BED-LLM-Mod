# Strategy-Prior Successor: DeepSeek V4 Flash Result

The distinct serving-verified successor does not pass the frozen L3 primacy gate.
DeepSeek V4 Flash first passed all ten strict no-repair L1/L3 serving cells, then ran
the unchanged 30-paired-trial, five-arm continuous L3 endpoint at seed 31003.

All required mechanics passed: legal actions, shared initial StrategyEIG/shared-d1
cells, matched width/grid scorer units, zero rollout LLM calls, and zero terminal cell
failures. The endpoint used 1,595 physical requests, including 1,563 accepted cells and
32 bounded raw rejects, with zero forced exits and `$0.8147969154` actual cost.

StrategyEIG beat width and matched grid-d2, but the claim requires strictly positive
paired lower confidence bounds against both random plans and shared-d1. Those intervals
were respectively `[-2.2959e-32, +0.0046131]` and
`[-2.2265e-32, +0.00002077]`; both fail the registered rule. The original L3 negative
result therefore remains unchanged, and this serving-verified successor makes the
negative conclusion robust to the prior Gemma reasoning-interface limitation.

Canonical artifact:
`copex_strategy_l3_successor_deepseek_v4_flash_high8192/20260718/L3.json`.
