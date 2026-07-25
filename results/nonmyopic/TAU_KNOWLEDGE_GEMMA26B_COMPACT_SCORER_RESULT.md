# tau-Knowledge Gemma 4 26B Compact Thinking Result

Date: 2026-07-25

## Decision

**The compact serving smoke failed closed, so the 20-task confirmation was not
run.** All logical calls completed and the semantic scores are diagnostically
strong, but two focused responses contain invalid JSON numeric literals. The
frozen parser rejects them and compact Gemma V2 is closed without repair.

Public failure artifact:

`results/nonmyopic/tau_knowledge_gemma26b_compact_scorer_smoke/tau-knowledge-gemma26b-compact-smoke-20260725T045442Z/SERVING_SMOKE_FAILURE.json`

## Exact Execution

| Metric | Result |
|---|---:|
| Logical prompts completed | 14 / 14 |
| Physical requests / HTTP attempts | 18 / 18 |
| Transport retries | 0 |
| Forced thinking exits | 4 |
| Forced-final requests / successes | 4 / 4 |
| Reasoning tokens | 65,173 |
| Prompt / completion tokens | 68,755 / 82,446 |
| Cost | $0.03621335 |

The compact output contract solved the original partial-object failure: every
length stop used the registered reasoning-disabled continuation, and all four
continuations returned a short answer.

## Serving Failure

Twelve of fourteen objects satisfy the frozen compact parser. Two focused
objects contain unquoted leading-zero literals (`05`, and one also contains
`08`). These are unambiguous to a person but invalid JSON and are not among the
registered integer or digit-string representations. The first strict decode
fails at character 21.

No literal was quoted, normalized, repaired, or reissued. No prompt, parser,
budget, task, score band, or gate changed after the responses.

## Diagnostic Only

For failure localization only, interpreting the invalid leading-zero literals
as their ordinary integer values gives:

| Signal | Diagnostic value | Frozen threshold |
|---|---:|---:|
| Focused pairwise accuracy | `.6765` | at least `.55` |
| Oracle-optimal focused choices | `8/10` | at least `7/10` |
| Focused mean regret | `.20` documents | descriptive |
| Focused vectors varying | `10/10` | at least `8/10` |

Under that non-protocol interpretation, every semantic efficacy gate passes.
This is useful evidence that Gemma 4 26B thinking can rank the frozen
continuations; it is not a passed serving smoke, policy result, or
cross-model confirmation.

## Consequence

The first Gemma interface had valid-row focused accuracy `.6538`; compact V2
has diagnostic accuracy `.6765` and `8/10` optimal choices. The repeated
semantic signal is encouraging, but neither preregistered interface completed
its exact serving conjunction. No Gemma V3 or 140-call confirmation follows.

The supported headline remains GPT-5.4 semantic ranking over generated
retrieval trees. Cross-model transfer remains suggestive rather than
confirmed.
