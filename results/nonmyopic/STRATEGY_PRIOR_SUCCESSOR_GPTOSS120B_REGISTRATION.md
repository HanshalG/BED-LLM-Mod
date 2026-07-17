# Strategy-Prior Successor: gpt-oss-120B Serving Registration

Registered on 2026-07-18 after two stronger hosted reasoning endpoints failed to emit a
final JSON cell within their respective smoke limits. This is the permitted serving
model swap within the distinct successor capability probe; the original L3 result is
unchanged.

`openai/gpt-oss-120b` is selected because OpenRouter exposes its native
`reasoning_effort` interface through the existing adapter, avoiding the
reasoning-only/empty-final behavior observed for Qwen 397B and DeepSeek V4 Pro. The
smoke uses high reasoning effort and a 4,096-token total output allowance. All ten
cells must still parse on the first response with zero forced exits.

At the completed L3's 1,373 physical-call reference volume, the 4,096-token maximum
and catalog completion price of `$0.17/M` give approximately `$0.96`; original prompt
volume at `$0.037/M` adds approximately `$0.02`. The formal run's `$3.00` hard cap is
therefore conservative. The formal endpoint remains the frozen paired L3 comparison
and is launched only after the serving gate passes.
