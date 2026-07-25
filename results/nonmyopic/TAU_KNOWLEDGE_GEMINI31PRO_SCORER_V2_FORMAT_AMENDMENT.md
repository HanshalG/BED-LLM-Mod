# Gemini 3.1 Pro Scorer V2 Format-Only Amendment

## V1 Failure

The frozen V1 smoke stopped after four of 14 calls. Both myopic root responses
returned complete parseable JSON. Both non-myopic full-tree responses hit the
4,096-token completion limit and returned truncated JSON. OpenRouter reported
12,155 reasoning tokens and two forced length exits, independently failing the
zero-reasoning gate. V1 cost `$0.202446`; no focused or confirmation call ran.

## Implementation Defect

The semantic tau V3.1 runner constructs its scorer through
`scripts.tau_knowledge_retrieval_opportunity._build_model`. That builder sends
an explicit `reasoning_effort="none"` OpenRouter control.

The generic cross-model harness instead called `build_model_adapter` directly.
Although the config requested no reasoning and the preregistration described
Gemini's explicit reasoning controls, the resulting request omitted the control
rather than explicitly disabling reasoning. Gemini then used most completion
tokens as internal reasoning.

## Frozen V2 Change

Before any V2 response, change only the generic cross-model model construction
to the original tau `_build_model(config)` path. This explicitly disables
reasoning and makes model construction match the original GPT-5.4 scorer.

Everything else is unchanged:

- same Gemini model, temperature, 4,096-token output allowance, and seed;
- same hash-locked two smoke trees and nonsemantic analysis;
- same 14-call smoke and conditional 140-call confirmation;
- same messages, parsers, gates, budgets, and private raw-response handling;
- no V1 response reuse, repair, continuation, or replacement; and
- no confirmation tree is scored unless V2 smoke passes every frozen gate.

If V2 reports any reasoning token, truncates, fails parsing, or misses an
efficacy gate, the Gemini line closes without V3.
