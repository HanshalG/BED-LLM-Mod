# Detective CA-BED Gemma Concise Shared-Tree Preregistration

Date frozen: 2026-07-24

## Motivation

The published-prompt DeepSeek serving interface failed twice before any
follow-up, hidden-role answer, or scientific endpoint. At 512 tokens it spent
the output on explanation and omitted every likelihood row. At 2,048 tokens,
one response omitted all required markers and another omitted one suspect.
Those runs are permanently serving-invalid and are not reused.

This is a distinct, pre-response interface qualification. It preserves the
LLM-native scientific question while replacing verbose explanation-followed-by-
format prompts with concise, output-only prompts already shown to be reliable
for Gemma 4 in this project.

## Frozen Design

- Dataset, hash, split seed, smoke cases, 12 ranking cases, and 24 untouched
  confirmation cases are identical to
  `DETECTIVE_CABED_DEPTH_RANKING_PREREGISTRATION.md`.
- Questioner, textual semantic likelihood estimator, and hidden-role answerer:
  `google/gemma-4-26b-a4b-it`, non-thinking.
- Temperature `0.0` for question generation, likelihood estimation, and
  hidden-role answers.
- Prompts request no explanation:
  - question generation must return exactly three
    `##Question##: [Target: Name] ...?` rows;
  - likelihood estimation must return exactly one
    `##Suspect Name##: probability` row for each of the four suspects;
  - answers must return exactly one `##Answer##: Yes|No` row.
- Output ceiling: 512 tokens. No content retry or parser repair.
- Uniform four-suspect prior, root/follow-up width `3/3`, confidence smoothing
  `0.7`, exact Bayes updates, depth-one immediate EIG, and depth-two expected
  total entropy reduction are unchanged.
- Each root retains four independently requested two-turn answer rollouts.
  Temperature zero makes these primarily a simulator-stability audit; their
  mean truth-log-posterior gain remains the realized root utility.
- Every tree and realized answer is shared across depth-one, depth-two, and
  seeded-random controls.

## Gates

The serving smoke retains the exact 104-request mechanics gates from the
original preregistration, including zero reasoning tokens and forced exits.
The 12-case ranking stage retains all original 624-request rank-fidelity,
paired truth-NLL, random-control, and entropy gates without alteration.

Smoke projected cost is `$0.10` with a `$0.50` hard cap. Ranking projected cost
is `$0.60` with a `$2.00` hard cap. The live OpenRouter balance is checked
before each paid stage. A smoke failure closes this concise Gemma interface.
