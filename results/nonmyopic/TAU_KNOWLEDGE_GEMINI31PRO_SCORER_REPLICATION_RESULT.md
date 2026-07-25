# tau-Knowledge Gemini 3.1 Pro Scorer Replication Result

## Decision

The scorer replication failed closed at serving and no confirmation ran. This
is a provider-interface failure, not an efficacy result.

## V1

V1 omitted a reasoning parameter, intending that reasoning was not requested.
Four calls completed:

- both myopic root responses returned complete parseable JSON;
- both non-myopic full-tree responses reached the 4,096-token limit and
  returned truncated JSON;
- OpenRouter reported 12,155 reasoning tokens and two forced exits; and
- cost was `$0.202446`.

The parser and zero-reasoning gates independently failed before any focused
continuation call.

An audit then found that the generic cross-model harness constructed the adapter
directly, unlike the original tau runner's explicit
`reasoning_effort="none"` builder. A format-only V2 amendment was frozen before
new responses.

## V2

V2 changed only model construction to the original tau explicit non-reasoning
path. OpenRouter rejected the request before generation with HTTP 400:
reasoning is mandatory for this endpoint and cannot be disabled.

V2 usage was zero physical requests, zero tokens, and `$0`. No response, score,
or endpoint was produced.

## Interpretation

Gemini 3.1 Pro cannot satisfy this replication's frozen zero-reasoning interface
through the current OpenRouter endpoint. Leaving reasoning implicit truncates
the long full-tree output; explicitly disabling it is unsupported. Increasing
the completion budget, permitting reasoning, changing prompts, or selecting a
third interface would define a new post hoc protocol. Per the V2 amendment, no
V3 was run.

The result provides no evidence for or against cross-family semantic scorer
transfer. The GPT-5.4 held-out result, nonsemantic-control result, and
refreshed-belief alignment null are unchanged.

## Usage and Artifacts

- V1 physical requests: 4.
- V1 cost: `$0.202446`.
- V2 physical requests and cost: 0 and `$0`.
- Confirmation requests: 0.
- Preregistration:
  `results/nonmyopic/TAU_KNOWLEDGE_GEMINI31PRO_SCORER_REPLICATION_PREREGISTRATION.md`
- V2 amendment:
  `results/nonmyopic/TAU_KNOWLEDGE_GEMINI31PRO_SCORER_V2_FORMAT_AMENDMENT.md`
- V1 failure:
  `results/nonmyopic/tau_knowledge_gemini31pro_scorer_smoke/tau-knowledge-gemini31pro-scorer-smoke-20260725T021215Z/SERVING_SMOKE_FAILURE.json`
- V2 failure:
  `results/nonmyopic/tau_knowledge_gemini31pro_scorer_v2_smoke/tau-knowledge-gemini31pro-scorer-v2-smoke-20260725T021517Z/SERVING_SMOKE_FAILURE.json`
- V1 raw responses: private and untracked.
