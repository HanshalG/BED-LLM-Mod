# tau-Knowledge Future-Subtree Alignment Ablation Result

Date completed: 2026-07-25

## Decision

**The serving smoke passed, but the causal confirmation failed one frozen
criterion.**

Correct root-to-future assignment improved aggregate ranking and endpoint
coverage relative to a within-response derangement. However, it lost on four
task endpoints, above the preregistered maximum of two. The result is
directional mechanism evidence, not a passed causal confirmation or a safety
claim.

## Intervention

Within each task, seed `24361` deranged the complete future subtree attached to
each of five roots:

- refreshed information-need hypotheses;
- four followup queries; and
- all followup retrieved documents.

The customer opening, initial hypotheses, root query, first results, root
position, and complete future-subtree multiset were preserved. Every
permutation was a derangement, and records with future fields removed had
identical canonical hashes.

Correct and deranged trees appeared as anonymous condition A/B in the same
model response. Seed `24362` assigned the correct tree to A on exactly 10/20
confirmation tasks. This removes separate-run OpenRouter drift from the
treatment comparison.

## Serving Smoke

The two-task mechanics smoke passed every frozen gate:

| Metric | Result |
|---|---:|
| Logical / physical / HTTP requests | 2 / 2 / 2 |
| Retries / reasoning / forced exits | 0 / 0 / 0 |
| Correct-tree A/B assignment | 1 / 1 |
| Varying correct / deranged score vectors | 2 / 2 |
| Cost | `$0.083740` |

Smoke efficacy was explicitly non-gating and adverse: pairwise accuracy was
`.75` correct versus `1.00` deranged, and selected-root oracle-tail coverage
was 2 versus 4.

## Confirmation

The confirmation completed exactly 20 paired requests with zero retries,
reasoning tokens, forced exits, parse failures, or tree-regeneration calls.

| Metric | Correct alignment | Deranged futures | Existing myopic |
|---|---:|---:|---:|
| Root pairwise accuracy | `.6529` | `.5124` | `.5537` |
| Pairwise points / comparable pairs | `79 / 121` | `62 / 121` | `67 / 121` |
| Oracle-optimal selected roots | `12/20` | `11/20` | `8/20` |
| Selected-root oracle-tail documents | `34` | `31` | `28` |
| Varying score vectors | `20/20` | `20/20` | `20/20` |

Correct alignment improved pairwise accuracy by `+.1405` over derangement and
`+.0992` over the existing myopic scorer. It improved selected-root oracle-tail
coverage by `+3` documents over derangement and `+6` over myopic. Correct and
deranged score vectors differed on all 20 tasks.

Task-level correct-minus-deranged outcomes were:

- endpoint wins / ties / losses: `4 / 12 / 4`;
- pairwise-accuracy one-sided sign-flip `p=.1536`;
- endpoint one-sided sign-flip `p=.3516`.

## Frozen Gates

All serving, intervention, blinding, variation, aggregate-accuracy, and
aggregate-endpoint gates passed. The only failed scientific criterion was:

- correct-alignment endpoint losses at most 2: observed 4.

Because every criterion was conjunctive, the overall status is `gate_failed`.
The loss cap is not relaxed after seeing the aggregate improvement.

## Interpretation

This intervention is more informative than the earlier belief-only
derangement. Moving refreshed belief text alone did not hurt performance,
because exact followup queries and documents remained attached to their
original roots. Moving the complete future subtree now reduces aggregate root
ranking substantially under a paired prompt, which is consistent with GPT-5.4
using semantic future consequences rather than only root priors.

The effect is not reliable across tasks. Four correct assignments choose roots
with lower oracle-tail coverage than their deranged counterparts, and neither
task-level test is significant. The defensible conclusion is therefore:

- correct semantic future assignment contributes useful aggregate signal;
- the signal is heterogeneous and does not satisfy the frozen safety criterion;
- this post hoc open-tree test does not establish a robust causal mechanism or
  add a new held-out/generalization result.

No alternate permutation, prompt, loss cap, ensemble, response reissue, or
additional confirmation follows.

## Cost And Artifacts

- Smoke cost: `$0.083740`.
- Confirmation cost: `$0.7957425`.
- Combined cost: `$0.8794825`.
- Project-ledger headroom after confirmation: `$19.710790376` above the
  protected `$25`.
- OatML use: none.
- Confirmation artifact SHA-256:
  `97d820fb59874370f9ea5a04992398af85503389753a46b9af44fba079b1e736`.
- Independent zero-call summary replay: passed exactly.

Artifacts:

- `results/nonmyopic/TAU_KNOWLEDGE_FUTURE_ALIGNMENT_ABLATION_PREREGISTRATION.md`
- `results/nonmyopic/tau_knowledge_future_alignment_smoke/tau-knowledge-future-alignment-smoke-20260725T054912Z/SERVING_SMOKE.json`
- `results/nonmyopic/tau_knowledge_future_alignment_confirmation/tau-knowledge-future-alignment-confirmation-20260725T055017Z/CONFIRMATION.json`
- `scripts/tau_knowledge_future_alignment_ablation.py`
- `tests/test_tau_knowledge_future_alignment_ablation.py`
