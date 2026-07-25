# ClariQ Native Two-Turn Opportunity Audit

## Source

- Official repository: `https://github.com/aliannejadi/ClariQ`
- Commit: `46885a544581a0af8aff0681d29e4971807e2912`
- `dev.tsv` SHA256:
  `68d2a5f87eab73721979b5f45f64099a9b2f080db1d0ce4b979d9daa4249906e`
- `question_bank.tsv` SHA256:
  `266f501bdc31afbc8a763685ea0de82949137a2779241b325f5ee03026f42f1b`
- `dev_synthetic.pkl.tar.gz` SHA256:
  `80abf68493d8f80b68121ded0c3557d18c13fe366bb4257670bbc586703af9f1`
- `multi_turn_dev_eval.pkl.tar.gz` SHA256:
  `86867d275242cd9f4ac17e62aa52ee34a13c351add6dbb2de746522397845381`

No model calls or cluster resources were used.

## Native Graph

ClariQ supplies:

- human-authored initial requests, facets, clarification questions, and answers;
- synthetic conversation contexts up to three turns;
- human question-bank actions;
- precomputed retrieval utility after asking each question and receiving its answer.

For each empty-history dev context, the audit reconstructed every first-question
transition by matching `(topic, facet, history + question/answer)` to its successor
context. Root immediate utility is the official `NDCG20.with_answer` value. Root
two-turn value is the maximum official `NDCG20.with_answer` among actions at the
matched successor. Greedy and depth-two roots use deterministic question-ID
tie-breaking.

## Result

| Metric | Value |
|---|---:|
| Reconstructed contexts | 15,345 |
| Usable initial contexts with at least two roots | 155 |
| Greedy and depth-two roots differ | 60 |
| Strict positive depth-two terminal gaps | 26 |
| Mean terminal gain over all usable contexts | `.002923` |
| Mean greedy terminal NDCG@20 | `.287603` |
| Mean depth-two terminal NDCG@20 | `.290526` |
| Maximum terminal gain | `.079478` |

Largest development opportunities include:

| Topic / facet | Initial request | Gain |
|---|---|---:|
| 44 / F0817 | Find me map of USA | `.07948` |
| 133 / F0138 | all men are created equal | `.06993` |
| 107 / F0035 | tell me about cass county missouri | `.05737` |
| 174 / F0295 | I want to learn about rock art. | `.03253` |
| 142 / F0171 | Find me information about the sales tax in Illinois. | `.03172` |
| 139 / F0156 | Tell me more about Rocky Mountain News | `.02614` |

## LLM-Native Construction

ClariQ avoids the autonomous action-diversity failure seen in AmbigDocs: the action
bank and realized answer transitions are externally authored. A future BED method can
still make the LLM irreducible by using it to:

1. maintain a semantic distribution over candidate facet descriptions;
2. map free-form clarification questions to `Y/N/U` response likelihoods over those
   facets;
3. update or regenerate semantic facet beliefs after the observed human answer; and
4. rank the externally grounded two-turn question tree.

The official retrieval NDCG is an external endpoint. Exact question-bank oracles,
compute-matched width, and random controls remain available.

## Caveat And Decision

The multi-turn graph is synthetic rather than a prospective live-user trajectory,
although its constituent questions, facets, and answers are human-authored and its
retrieval endpoint is externally precomputed. Development opportunity is not a
confirmation claim.

ClariQ passes the zero-cost structural gate. The next step is a target-free semantic
likelihood serving gate on an already-open positive-gap dev context. No endpoint
policy run is authorized until repeated likelihood stability and response-partition
diversity pass.
