# HotpotQA Train Future-Uplift Confirmation Preregistration

Frozen on 2026-07-26 before reading any training-split question, answer,
context, supporting fact, or policy endpoint.

## Purpose

Test a single score-calibration change motivated independently by two tau-
Knowledge blocks and the already-open Hotpot causal smoke. The old model-aware
total adds an immediate score and a continuation score that are not calibrated
to a common scale. The new policy selects the largest isolated future term,
which is exactly `full-tree score - isolated myopic score` in this interface.

On the old public smoke this rule would select the structurally enabling root:
future scores were `56/100/100/99`, and its frozen tie-break uses immediate
score then root order. This is development evidence only. The present study
uses previously untouched HotpotQA distractor-training records.

## Source And Metadata-Only Split

Official Hugging Face converted Parquet revision:
`refs/convert/parquet`, configuration `distractor`, split `train`.

- shard 0 SHA-256:
  `76d3bb3048a7cc73c1958107c0c5872a00d7e7d00c105b81e92f6769e7822e68`,
  45,224 rows;
- shard 1 SHA-256:
  `713661628434fbb19fff7392e2e321e4ed107e3c7c7784d0690946e5f722763f`,
  45,223 rows;
- total 90,447 rows, including 72,991 bridge rows;
- metadata-only SHA-256 over `id/type/level`:
  `e75f2c4ec26b8e3edefb76a1efa81390d9440b634c435ab3a978b5e455c83542`.

Sort bridge IDs, shuffle with `random.Random(24419)`, then freeze:

| Split | Count | Ordered-ID SHA-256 |
|---|---:|---|
| opportunity | 1,000 | `2e63446b32879cae47876214261054ef696e153ebf621042fc5420bc7e1eb60c` |
| development | 100 | `01755657af1915c4c3171cbc2454f5c12ea6e48e18f8c5fae7cf7fbb68db3a7a` |
| confirmation cohort | 500 | `eb97d88de626bcf6aefa118f0ec19e2d62200495200da4ed63dc5587aa756079` |
| holdout | 71,391 | `f1fc6d1119a8fdda01860055adf1d146f900433c76f1af508bc725ff4ec088d8` |

Endpoint columns are materialized only through an Arrow ID predicate for the
active split. The development and holdout endpoint columns remain sealed.

## Zero-Call Opportunity Gate

Apply the exact validation-split directional-unlock rule to the 1,000
opportunity rows. A row qualifies for the policy test when:

1. it is a strict one-way enabling-to-answer support link;
2. both support documents occur among the title-BM25 top four; and
3. the title-BM25 top root is not the enabling document.

All must pass:

- at least 250 strict directional unlocks;
- at least 20 qualifying top-four misses;
- every qualifying row has structural enabling-first gain exactly one;
- exactly 1,000 opportunity endpoint rows and zero endpoint rows from every
  other split are materialized;
- zero model calls and zero cost.

Failure closes the training route.

## Model And Shared Tree

Model: `openai/gpt-5.4`, temperature zero, explicit non-reasoning.

Select the first ten qualifying rows in frozen confirmation-cohort order.
Fewer than ten fails before model construction. For each task:

1. one initial call generates eight answer/support-chain hypotheses and four
   immediate root scores;
2. four root-paragraph calls regenerate eight unresolved hypotheses;
3. four state-only calls score all nine remaining titles under aligned,
   cyclic-shuffled, and unchanged-initial beliefs in one response;
4. one final call answers from the two documents selected by future uplift.

Thus confirmation is exactly 100 calls. All responses are checkpointed before
parsing the completed stage. The model never receives answers, supporting
facts, support roles, endpoint values, or the cohort selection rule.

## Frozen Policies

Every policy shares roots, revealed paragraphs, refreshed beliefs, title
scores, and common random numbers.

- **Future uplift:** maximize the best aligned continuation score; break ties
  by immediate score, then root order. Follow the aligned argmax.
- **Myopic receding:** maximize immediate score; follow the aligned argmax.
- **Old total:** maximize immediate plus best aligned continuation; follow the
  aligned argmax.
- **Fixed:** maximize immediate plus unchanged-belief continuation; follow
  that continuation.
- **Shuffled:** use the cyclic wrong branch belief.
- **Random receding:** seeded root (`24420 + task index`), aligned continuation.

Exact endpoint is unique annotated support-document coverage after two
documents. Secondary endpoint is token F1 of the future-uplift final answer.
For root-ranking fidelity, each root's realized value is coverage after its own
aligned best continuation.

## Confirmation Gates

All mechanics conditions and all scientific conditions in
`scripts/hotpot_future_uplift_confirmation.py` are conjunctive. Central
scientific thresholds are frozen as:

- at least 4/10 root changes from myopic;
- at least 5/10 enabling-root selections;
- at least 8/10 enabling branches select the answer document;
- future uplift covers at least 18/20 supports;
- gain at least two supports, wins greater than losses, and one-sided exact
  sign-flip `p <= .10` versus myopic;
- gain at least two and wins greater than losses versus old total;
- gain at least two and wins greater than losses versus seeded random;
- future-score root-pair accuracy at least `.60` and at least `.05` above
  immediate-score accuracy;
- mean final-answer token F1 at least `.40`;
- exact 100 requests/HTTP, zero retry/reasoning/forced/malformed responses,
  strong branch-state/vector variation, and cost at most `$1.20`.

No threshold, coefficient, cohort size, parser, prompt, or task may be changed
after opportunity endpoints or model responses are viewed. There is no retry,
repair, replacement, subset result, or confirmation rerun.

## Serving And Budget

Before confirmation, run one exact ten-call all-stage serving pass on the
already-open validation smoke task. It tests transport and parser mechanics
only and cannot authorize a scientific claim. Serving cap is `$0.15`.

Combined hard cap is `$1.35`, below the frozen `$1.77849905` research
allowance. The authenticated account reports `$34.793005094` remaining; at
least `$25` stays protected through Monday. OpenRouter only; no OatML or
cluster.
