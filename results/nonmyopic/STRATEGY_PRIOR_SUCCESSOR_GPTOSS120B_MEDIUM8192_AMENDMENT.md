# gpt-oss-120B Serving Configuration Amendment

Registered on 2026-07-18 before the corrected gpt-oss smoke. The first gpt-oss attempt
did not test a viable serving configuration because high reasoning consumed its entire
4,096-token allowance. This amendment changes only the serving allocation, not the
scientific endpoint, controls, seed, or pass rule.

The corrected configuration uses `reasoning_effort: medium` and `max_tokens: 8192`.
OpenRouter documents medium effort as allocating approximately half of `max_tokens` to
reasoning, leaving approximately 4,096 tokens for the compact JSON cell. The model is
still reasoning-enabled and must pass the same ten no-repair, zero-forced-exit gate.

At the completed L3's 1,373-call reference volume, the absolute 8,192-token output
maximum costs approximately `$1.91` at gpt-oss-120B's `$0.17/M` completion price, plus
approximately `$0.02` for the historical prompt volume. The successor run retains its
independent `$3.00` ledger cap.

Official reasoning allocation reference:
<https://openrouter.ai/docs/guides/best-practices/reasoning-tokens>.
