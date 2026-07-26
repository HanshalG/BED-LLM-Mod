# VoI Medical Coherent Future-Tree Mechanics Result

## Decision

The preregistered mechanics gate failed closed at response-model coherence.
This run does not score the tree and provides no evidence for or against a
depth-two policy advantage.

The exact V1 interface is closed. It must not be repaired, rerun with another
seed, or analyzed as a partial tree.

## Frozen Run

- Preregistration and implementation commit: `db27b08`
- Run ID:
  `voi-medical-future-tree-mechanics-20260726T060126Z`
- Interface: `voi-medical-future-tree-mechanics-1`
- Model: `openai/gpt-5.4` through OpenRouter
- Reasoning: disabled
- Source SHA-256:
  `e851864a9cb53c36304245bc3213a8a894cf7b86f8945d60978923e1f1ef0169`
- Public failure SHA-256:
  `2cf6e09ec51de7da3fbb128bae64bf48f58441823d21de07408de961ad2d46b2`
- Private raw SHA-256:
  `0d739d6503cf039ffbe64f0a532dab94b987a007b55e963123e9a12d4ed9b12d`

## Execution

All frozen model requests completed:

- physical requests: `18`
- HTTP attempts: `18`
- retries: `0`
- reasoning tokens: `0`
- forced exits: `0`
- forced-final requests: `0`
- prompt tokens: `3,325`
- completion tokens: `4,299`
- cost: `$0.0727975`

The four roots, root answer matrix, twelve branch-specific two-question sets,
and four root-local follow-up answer matrices all reached the strict parser.
The run then failed while merging the parsed follow-up likelihood maps.

## Failure

The same normalized semantic question appeared under more than one tree
context, but its deterministic diagnosis label was not stable:

- `Do you have heartburn?` assigned Pancreatitis `No` in the second root's
  matrix and `Maybe` in the third root's matrix.

The preregistration explicitly required duplicated likelihood cells to be
consistent. The implementation therefore raised
`duplicate follow-up has inconsistent answer maps` before any entropy score,
root selection, hidden patient, or endpoint was computed.

An audit-only comparison of all parsed maps, including roots, found:

- `28` generated action occurrences;
- `21` unique normalized questions;
- `7` repeated occurrences;
- `2` inconsistent repeats, each differing on one diagnosis;
- the second inconsistency was `Do you have a cough?`, where Esophagitis
  changed from `Maybe` as a root to `No` in a follow-up matrix.

These are response-model coherence diagnostics, not partial scientific scores.

## Interpretation

The failure is not a transport, parser-shape, token-budget, or reasoning
failure. It exposes a modeling defect in root-local classification: independently
classifying the same semantic action in different question batches permits
surrounding context to change the purportedly frozen likelihood function.

A valid exact tree needs one globally coherent response map for each unique
semantic question. The only scientifically defensible successor would be a
separately preregistered interface that globally deduplicates generated
questions before classification and classifies every unique question exactly
once. This result does not authorize that successor automatically.

## Budget

The run spent `$0.0727975`, leaving:

- frozen research allowance: `$1.20891155`;
- live OpenRouter balance: `$34.223417594`;
- balance above the protected `$25` Monday reserve: `$9.223417594`.

OatML and cluster jobs: `0`.
