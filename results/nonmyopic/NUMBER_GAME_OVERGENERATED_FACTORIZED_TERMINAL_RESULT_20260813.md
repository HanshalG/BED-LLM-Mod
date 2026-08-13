# Number Game Overgenerated Factorized Terminal Result

The exact fresh interface at pushed commit `96dc05c7` is terminally closed.

All 40 Qwen3.7-plus description-only requests returned strict JSON with clean `stop`
finishes, zero reasoning, and zero retries. Item-isolated replay retained 302 of 320
descriptions: 4 failed the lexical gate and 14 used prohibited observed-answer
language. Nine draws retained 29--32 valid descriptions, but one draw retained 24
because one shard had zero of eight valid items. That violates the prospectively
frozen minimum of seven valid items per shard and 28 per draw.

The failure occurred before adapter construction for the 30 history-blind DeepSeek
translations and 10 mask/history-blind Luna audits. Translation, semantic audit,
target, and endpoint counts are all zero. The exact cost is `$0.020880640`; fully
posted cumulative OpenRouter usage is `$220.334124806`, so Aug 13 account-wide spend
against the frozen `$220.134128880` boundary is `$0.199995926`.

This is a proposal-mechanics null, not evidence about translation fidelity, semantic
support quality, non-myopic planning, or endpoint efficacy. The strong aggregate does
not rescue the failed per-shard gate. Do not rerun, change the threshold, reuse these
descriptions, or open any scientific descendant from this interface.
