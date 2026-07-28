# Semantic Object Game Depth-Three Mechanics Preregistration

Date: 2026-07-29

## Question

Can the cross-fitted depth-three method that worked in the Number Game be
executed in a distinct semantic concept-identification task where the LLM is
load-bearing?

This is a one-tree mechanics and opportunity gate. It cannot establish an
efficacy claim.

## LLM-Native Environment

The fixed universe contains 32 named real-world objects spanning food, animals,
plants, tools, transport, instruments, devices, and household objects. A
hypothesis is:

1. a natural-language description of one coherent semantic property; and
2. the complete set of universe objects satisfying that property.

GPT-5.4 Mini generates the initial support and regenerates it after every
hypothetical observation. The complete membership set supplied with each
hypothesis is its frozen extension, so the LLM jointly supplies the open
semantic hypothesis and its answer likelihood table. A parser rejects invalid,
duplicate, constant-like, or observation-inconsistent extensions. Consistent
parent hypotheses are retained at each refresh.

This is not an executable arithmetic-rule task. Removing the LLM would remove
the semantic hypotheses and membership relations on which planning operates.

## Frozen Tree

- planning model: `openai/gpt-5.4-mini`;
- independent target model: `google/gemini-2.5-flash`;
- reasoning: disabled;
- temperature: `0.7`;
- planning seed: `36000`;
- validation seeds: `36100--36103`;
- endpoint seeds: `36200--36207`;
- proposals per support: `16`;
- candidate roots: `6`;
- validation draws: `4`;
- endpoint draws: `8`;
- support mode after both observations: generated hypotheses plus all
  observation-consistent parent hypotheses, deduplicated by full extension.

The candidate roots include the myopic-EIG root, the fixed-support depth-two
root, distinct high-EIG signatures, and seeded remaining roots. Every root has
both first-answer supports. Each first-answer support chooses its exact best
second query, and both second-answer supports are regenerated.

Depth two and depth three are selected by mean posterior-predictive Brier risk
over all hypotheses in the four independent validation draws. The endpoint
uses all hypotheses in the eight separate target draws. Targets never affect
tree generation or root selection.

## Exact Calls

The gate requires exactly 49 accepted requests:

- one initial planning support;
- 12 first-answer supports;
- 24 second-answer supports;
- four validation supports; and
- eight endpoint supports.

All responses use strict structured output. Logged transport retries count
against HTTP attempts. There is no semantic repair, normalization, response
reissue, dropped support, or partial result.

## Frozen Gates

Every condition must pass:

1. exact 49 accepted requests and exact transport accounting;
2. zero reasoning tokens and zero forced exits;
3. total cost at most `$0.75`;
4. at least 12 valid unique initial concepts;
5. at least six retained unique concepts in every first branch;
6. at least four retained unique concepts in every second branch;
7. at least 12 valid unique concepts in every validation and endpoint draw;
8. six finite depth-two and depth-three root risks;
9. nonzero depth-three risk range; and
10. depth two and depth three select different roots.

The descriptive endpoint Brier, Hamming error, and truth-extension coverage
are recorded regardless of direction. They do not gate or justify a claim.

## Decision Rule

A complete pass authorizes only a separately preregistered small multi-tree
development comparison with paired depth-three, equally cross-fitted depth-two,
myopic, fixed-support, and seeded-random controls. Its thresholds, seeds,
sample size, confidence interval, and cost cap must be frozen before any new
response.

Any failed gate leaves this exact interface closed. A transport-distinct
successor is admissible only for a clearly diagnosed serving failure and must
be frozen before new responses; endpoint direction cannot motivate a repair.

## Budget And Integrity

The user reports a `$40` top-up and requests that the total budget be paced
over four days without a reserve. Immediately before preregistration, the
authenticated credits endpoint still reports `$180.00` total and
`$161.579636306` used, leaving `$18.420363694` visibly available. This gate is
within both the visible balance and the approximate four-day pace.

OpenRouter only. OatML, Slurm, SSH, and cluster resources are not used.
