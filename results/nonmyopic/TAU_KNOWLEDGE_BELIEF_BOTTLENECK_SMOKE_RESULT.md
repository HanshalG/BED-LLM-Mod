# tau-Knowledge Paired Belief-Bottleneck Smoke Result

## Decision

The frozen smoke failed closed. GPT-5.4 returned 18 unambiguous but
noncanonical `"09"` scores across five responses, so only 5/10 responses
passed the preregistered parser. There is no parser amendment, repair,
replacement response, development stage, or policy run.

More importantly, a clearly labeled diagnostic that interprets `"09"` as 9
points opposite the hypothesis: correctly aligned generated beliefs rank the
same evidence worse than cyclically shuffled beliefs. The belief-bottleneck
route is therefore closed for these tau trees independently of the mechanical
format failure.

## Frozen Mechanics

- Source artifact SHA-256:
  `61c466ead49a5545abd54dd28e1a2fd7ad070287f5108684a3b2643befc40a01`.
- Rows: exact `10/10`, five roots from each of two public smoke tasks.
- Aligned/shuffled state pairs distinct: `10/10`.
- Blinded labels: aligned=A on `5/10`, aligned=B on `5/10`.
- Successful OpenRouter responses: `10`.
- HTTP attempts: `10`; transport retries: `0`.
- Reasoning tokens: `0`; forced exits: `0`.
- JSON objects with exact keys: `10/10`.
- Canonically parsed objects: `5/10`.
- Noncanonical fields: `18`, all the exact string `"09"`.
- Repairs or scientific retries: `0`.
- Adapter-attributed cost: `$0.0688825`, below the `$0.25` cap.

The paired prompt hid the customer opening, initial beliefs, all query text,
first results, document IDs, BM25 scores, and required-document endpoints. It
showed only two blinded eight-hypothesis states and the identical four sets of
three document titles plus 500-character excerpts. Both states were scored in
the same physical response.

## Diagnostic Only

The following values are not a gate rescue. They are computed only because
every rejected field has one unambiguous integer interpretation.

| Metric | Aligned beliefs | Shuffled beliefs | Aligned minus shuffled |
|---|---:|---:|---:|
| Pairwise accuracy, 17 comparable pairs | .4412 | .7059 | -.2647 |
| Oracle-optimal choices, 10 roots | 7 | 8 | -1 |
| Selected exact required-document total | 5 | 6 | -1 |

Aligned scores varied on `10/10` roots, so score collapse does not explain the
result. Four roots had tied exact values across all candidate continuations and
therefore contributed no pairwise comparisons.

The frozen aligned thresholds required accuracy at least `.60`, an accuracy
gain of at least `+.10`, at least two additional optimal choices, and at least
two additional selected exact documents. The diagnostic misses every causal
gain threshold and reverses all three aligned-versus-shuffled contrasts.

## Interpretation

The earlier full-scorer shuffle could be criticized because opening, query,
and path evidence let the model bypass the refreshed beliefs. This paired
intervention removes those channels and controls provider nondeterminism
within each response. It still provides no evidence that the correctly
aligned generated information-need state improves continuation ranking.

This is only a two-task public smoke and cannot establish that generated
beliefs are generally harmful. It does establish that these generated states
do not clear the preregistered causal and efficacy gate needed to justify
old-tree development. The successful tau result remains evidence for
load-bearing semantic document scoring over LLM-generated trees, not for a
causal branch-to-belief update mechanism.

## Artifacts

- Preregistration:
  `results/nonmyopic/TAU_KNOWLEDGE_BELIEF_BOTTLENECK_SMOKE_PREREGISTRATION.md`
- Failed-closed public artifact:
  `results/nonmyopic/tau_knowledge_belief_bottleneck_smoke/tau-knowledge-belief-bottleneck-smoke-20260725T032521Z/SERVING_SMOKE_FAILURE.json`
- Private raw-response SHA-256:
  `b8a83fc98e9347d8706315a0d3b8f352207c3447215d7797ac42e6f44f7e4a3c`
