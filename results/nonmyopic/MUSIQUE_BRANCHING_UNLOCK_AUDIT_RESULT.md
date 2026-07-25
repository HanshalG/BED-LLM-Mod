# MuSiQue 4-Hop Branching Unlock Audit Result

Date: 2026-07-25

## Outcome

**Every frozen zero-call gate passed.** The fresh official train split contains
a large, externally annotated two-retrieval asymmetry suitable for a small
LLM-native mechanism smoke.

Artifact:

`results/nonmyopic/musique_branching_unlock_opportunity/AUDIT.json`

## Prospective Integrity

- Official MuSiQue-Ans train SHA-256:
  `83a75b1e11e4e9bb8f8308e72ac40ca617ae4431b3a0d955b61cab259248490a`.
- Exact source rows: 19,938.
- Exact `4hop3` rows: 400.
- Opportunity/development/holdout sizes: 120/40/240.
- All three ordered split hashes reproduced.
- Only the 120 opportunity endpoints were accessed.
- Model calls and cost: zero.

The earlier dev-split aggregate and eight inspected examples remain disclosed
development work and are not counted as evidence.

## Frozen Results

| Metric | Required | Observed |
|---|---:|---:|
| Exact branching DAG and four distinct supports | 120/120 | 120/120 |
| Deep-first minus shallow-first connected-prefix gap | 1 on all | 1 on all |
| Neither root answer stated in final question | at least 110 | 120 |
| Shallow-root BM25 score above deep-root | at least 45 | 52 |
| Deep root not BM25 rank one | at least 85 | 85 |
| BM25 top is shallow root or distractor | at least 65 | 65 |
| Neither root title stated in final question | descriptive | 49 |

BM25 top roles were 35 deep roots, 5 deep children, 25 shallow roots, 15
final supports, and 40 distractors. Mean deep-root rank was `4.583`; mean
shallow-root rank was `5.742`.

## Interpretation

The construction fixes the commutativity that erased the Hotpot result. A
two-action deep-first path can retrieve the deep root and its dependent child.
A shallow-first path can still retrieve two support documents, but the second
action can at best begin the deep chain, leaving no resolved dependency edge.

This does not yet show that an LLM can identify or value the deep root. Three
lexical thresholds passed exactly at their preregistered boundaries, so the
audit supports only the structural opportunity, not a broad lexical-bias
claim.

A separately frozen development smoke may now test whether GPT-5.4 generates
both roots, regenerates useful dependency beliefs after the deep paragraph,
and assigns enough continuation value for depth two to improve exact connected
prefix over a receding myopic policy. Gold decompositions cannot enter any
generator or scorer prompt. The 240-row holdout remains sealed.
