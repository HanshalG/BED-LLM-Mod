# tau-Knowledge V3.1 Nonsemantic Control Result

## Decision

The LLM semantic scorer passes every frozen load-bearing criterion against three
zero-call nonsemantic policies on the identical confirmation trees.

This is a post hoc reviewer diagnostic: the main confirmation endpoints were
already known before the control definitions were fixed. It strengthens the
mechanism interpretation but is not a new preregistered policy claim.

## Results

| Policy | Documents | Root accuracy | Continuation accuracy | Optimal | Regret |
|---|---:|---:|---:|---:|---:|
| LLM semantic V3.1 | 30 | 0.7025 | 0.7566 | 80/100 | 21 |
| Raw BM25 sum | 22 | 0.5372 | 0.6184 | 71/100 | 30 |
| Novel-document count | 21 | 0.4256 | 0.5395 | 69/100 | 35 |
| Lexical-IDF overlap | 17 | 0.5455 | 0.6316 | 75/100 | 27 |

Paired endpoint comparisons favored semantic V3.1:

- versus raw BM25: 9 wins, 2 losses, 9 ties, total `+8`;
- versus novel count: 9 wins, 3 losses, 8 ties, total `+9`;
- versus lexical IDF: 11 wins, 1 loss, 8 ties, total `+13`.

For every control, semantic V3.1 exceeded endpoint coverage by at least three
documents, had more wins than losses, and achieved higher root and continuation
pairwise accuracy.

## Interpretation

Tree width or raw retrieval strength does not explain the held-out result. The
same LLM-generated roots and branches perform substantially worse when scored by
BM25 magnitude, document novelty, or literal overlap with the generated beliefs.
The useful component is the model's semantic judgment that a returned policy
document supports a plausible unresolved need.

The control does not show that every LLM component is uniquely necessary. All
policies still consume GPT-5.4-generated openings, hypotheses, roots, and
followups. It isolates semantic branch scoring, not the generator, and remains
post hoc.

## Artifacts

- Frozen plan:
  `results/nonmyopic/TAU_KNOWLEDGE_V3_1_NONSEMANTIC_CONTROL_PLAN.md`
- Analysis:
  `results/nonmyopic/TAU_KNOWLEDGE_V3_1_NONSEMANTIC_CONTROL_ANALYSIS.json`
- Confirmation:
  `results/nonmyopic/tau_knowledge_receding_continuation_v3_1_confirmation/tau-knowledge-receding-v3-1-confirmation-20260725T030000Z/CONFIRMATION.json`
