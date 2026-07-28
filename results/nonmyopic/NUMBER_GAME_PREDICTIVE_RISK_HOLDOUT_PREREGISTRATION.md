# Number Game Predictive-Risk Holdout Preregistration

Date frozen: 2026-07-28, before the fresh target-model response.

## Frozen Development Source

This test uses the already-published Gemini proposal tree:

- `MODEL.json` SHA-256:
  `bf45eb9f5dd8da0289d834671952b53f9dff8fc7144190fe1ecfab50045bcecb`
- `RESULT.json` SHA-256:
  `6c1de697892c387df02ab0a2f657f05b1a2164872192653b407281e8e22a453c`

The failed max-future-EIG objective is not rescored as if it were successful.
This is a prospectively defined successor evaluated on fresh targets.

## Frozen Policy

For each candidate root and each current initial particle treated as the hidden
truth:

1. Use the already-generated support for the particle's root label.
2. Select the second query by greedy EIG on that regenerated support.
3. Apply the simulated truth's second label and filter the support.
4. Measure posterior-predictive Brier loss over all unqueried integers. Empty
   support receives loss one.

Average terminal loss uniformly over current particles and choose the root with
minimum loss, breaking ties by Hamming error, truth retention, then lower
integer. This frozen computation selects root 34. Myopic EIG and classical
fixed-support depth two both select root 48.

The selected root is not changed after seeing the fresh targets.

## Fresh Holdout

- Exactly one `openai/gpt-5.4` nonreasoning, temperature-zero, strict-schema
  call generates 24 executable target concepts from the same no-observation
  Number Game prompt.
- Parsing and filtering are unchanged. At least 16 valid unique targets and at
  least eight extensions absent from the Gemini planning support are required.
- Each policy receives the same frozen Gemini branch supports and executes two
  queries. The random control is the exact uniform average over all eight
  candidate roots, not one lucky draw.
- Primary metrics use all valid fresh targets. Novel-target metrics are a
  required transfer diagnostic.
- Paired Brier intervals use 20,000 target bootstraps with seed 26069.
- Exactly one accepted request, no reasoning tokens or forced exit, and cost at
  most `$0.20`.

## Pass Criteria

All must pass:

1. The frozen source computation reproduces root 34 and at least 10% Brier
   improvement over myopic root 48 on current particles.
2. On fresh GPT-5.4 targets, root 34 improves mean Brier and best-rule Hamming
   error by at least 5% versus myopic/fixed root 48.
3. The paired 95% bootstrap interval for candidate-minus-myopic Brier lies
   wholly below zero.
4. Root 34 has no exact-extension coverage loss versus root 48.
5. Root 34 improves mean Brier by at least 5% versus the uniform random-root
   control.
6. On target extensions novel to the planning support, both Brier and Hamming
   are directionally better than myopic.

Pass authorizes a separately frozen independent proposal-tree replication.
Failure closes this frozen root and terminal-risk rule on the source tree.
