# Qwen 397B Explicit-Reasoning-Budget Amendment

Registered on 2026-07-18 before a new Qwen serving gate. The first Qwen smoke sent
only `reasoning.enabled: true`, allowing the provider to spend all 768 output tokens
on reasoning and return no final content. That was a serving-allocation failure, not a
test of Qwen's strategy JSON quality.

The corrected request uses OpenRouter's explicit `reasoning.max_tokens: 512` with a
total `max_tokens: 768`, reserving 256 tokens for the final compact strategy cell.
This parameter is documented for compatible Alibaba/Qwen reasoning models. The ten
no-repair L1/L3 parser gate is repeated under this distinct serving configuration; no
policy endpoint may be read unless it passes.

The formal L3 configuration remains frozen. At 1,373 calls, the absolute 768-token
completion ceiling at Qwen's `$2.34/M` price plus the prior prompt volume is
approximately `$2.67`, within the independent `$3.00` cap.

Official API reference:
<https://openrouter.ai/docs/guides/best-practices/reasoning-tokens>.
