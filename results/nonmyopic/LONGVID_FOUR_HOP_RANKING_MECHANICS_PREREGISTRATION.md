# LongVidSearch Four-Hop Ranking Mechanics Preregistration

## Status

Frozen after the semantic-support smoke and before any mechanics response.
This is a disclosed-development ranking-fidelity gate. It cannot support a
reserve or headline claim.

## Bound Inputs

- four-hop opportunity audit SHA-256:
  `3cb882b1facbef2ababe2f8be68529423773c362a1eb4402b68d0bb240cc3ce0`;
- four-hop V2 confirmation SHA-256:
  `895e448c047ca7afa924393da0a4f637a21735c8a770336fa17db0a72fdd4081`;
- semantic-support smoke SHA-256:
  `0e594dbdaa748160e9006830992cca000ccfc7badec8cdde3b6d7ca56d412239`.

Use the first six strict tasks from the disclosed opportunity block. Their two
candidate roots are the frozen greedy/oracle pair, independently shuffled with
seed `270739`:

| Row | Blinded root order |
|---:|---|
| 955 | 5, 0 |
| 1802 | 2, 1 |
| 540 | 0, 9 |
| 479 | 5, 4 |
| 1332 | 2, 3 |
| 1068 | 3, 13 |

Layout hash:
`6beaa5bccbe31aac2eedb44cf73ea2c3a6f11846c88012d325bb3c0247fbd3a3`.

The model receives only the question, candidate query, retrieved captions, and
its own prior support. It never receives greedy/oracle labels, answer text,
evidence slice IDs, path coverage, or any endpoint-derived value.

## Policy Mechanics

Model: `openai/gpt-5.4`, nonreasoning, temperature `0`, zero retries.

For each task:

1. generate one six-particle initial semantic support;
2. execute each of the two blinded candidate roots with deterministic BM25;
3. after every retrieved caption, regenerate the six-particle support;
4. for steps 2--4, execute the search query from the highest-weight particle;
5. exclude every previously retrieved caption; and
6. stop after four distinct retrieved captions.

This requires exactly:

- 6 initial-support calls; and
- 48 refresh calls: 6 tasks x 2 roots x 4 observations.

Total: 54 physical requests and HTTP attempts.

The exact smoke grammar, six particles, anchor validation, raw-before-parse
checkpointing, and no-repair rules are unchanged.

## Model-Native BED Score

Normalize each support's six integer weights and compute Shannon entropy in
nats.

- Immediate score: initial entropy minus entropy after the first caption.
- Final non-myopic score: initial entropy minus entropy after the fourth
  caption.

The model-native depth-one policy selects the candidate with larger immediate
score. The depth-four policy selects larger final score. Candidate-order ties
select index 0. A seeded random control uses seed `270740`.

This score intentionally evaluates the LLM's own path-dependent belief
dynamics. Supports need not share particle identities across turns; confidence
concentration after regeneration is the state variable.

## Delayed Endpoint

The six hidden four-clip evidence sets are not loaded until:

- every model response is checkpointed and parsed;
- every four-step query path is complete;
- all immediate/final scores and choices are frozen; and
- a pre-endpoint checkpoint is written.

Then report the number of necessary clips among each candidate's four
retrieved captions. No model call occurs after endpoint load.

## Frozen Gates

All must pass:

- exact 54 physical requests and 54 HTTP attempts;
- zero retries, reasoning tokens, and forced exits;
- all 54 supports parse;
- all 12 paths contain four distinct captions;
- at least 44/48 refresh supports differ from their preceding support;
- all 48 refreshes have at least four valid observation anchors;
- at least 4/6 tasks have different realized candidate coverage;
- at least 4/6 tasks have final-score separation of `.02` nats;
- final pairwise accuracy among rankable tasks is at least `.75`;
- final pairwise accuracy strictly exceeds immediate accuracy;
- final-score Spearman correlation with candidate coverage is at least `.25`;
- final-score Spearman strictly exceeds immediate-score Spearman;
- final-selected total coverage is at least immediate-selected coverage;
- final-selected total coverage strictly exceeds seeded-random coverage; and
- adapter cost is at most `$0.75`.

Failure closes this exact entropy-ranking route. Passing authorizes only a
separately frozen root-generation development gate before any reserve-video
policy.

## Budget

Projected cost: `$0.35`; hard cap: `$0.75`. The conservative balance before
this stage is `$33.519207594`, with `$25` protected through Monday and
`$8.445165` remaining under the pre-Monday ceiling. OpenRouter only; no
OatML/Slurm.

