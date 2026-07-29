# Number Game Qwen First-Link Confirmation-64 Preregistration

Date frozen: 2026-07-29, before any response from smoke seed `60000` or
confirmation seeds `60100..61611`.

## Question

On 64 wholly fresh LLM-generated belief trees, does the simulated
path-dependent depth-three advantage over the myopic root predict the same
root pair's exact-canonical realized Brier advantage?

This prospectively tests the first link that was identified retrospectively
across the prior four 32-tree blocks. It does not test online execution or a
second noisy posterior-update link.

## Serving Gate

Before the confirmation, make exactly ten non-reasoning
`qwen/qwen3.7-plus` requests with the existing Number Game proposal schema:
two initial histories, four one-observation histories, and four
two-observation histories.

All ten must parse, both initial supports must contain at least 16 valid
unique rules, every conditioned support at least eight, and accounting must
show exactly ten accepted HTTP attempts, zero retries, reasoning tokens, and
forced exits. Hard smoke cap: `$0.10`.

Failure stops the confirmation. There is no repair or reissue.

## Frozen Confirmation

- Planner: `qwen/qwen3.7-plus`, non-reasoning, temperature `0.7`.
- Validation generator: `google/gemini-2.5-flash`, non-reasoning,
  temperature `0.7`.
- Tree seeds: `60100..60163`.
- Mechanics-only target seeds: `60200..60263`.
- Eight validation supports per tree: seeds `61100..61611`.
- Retained-rejuvenation at both answer-conditioned refreshes.
- Root selection uses only the eight validation supports.
- Endpoint: all 33 Tenenbaum--Griffiths concepts, equal concept and tree
  weight, exact extensions over `0..100`.
- Generated mechanics targets never enter efficacy or root selection.
- Bootstrap: 20,000 tree resamples, seed `61700`.

## Primary First-Link Gates

On trees where depth three and myopic EIG select different roots, all must
hold:

- at least 56 changed roots;
- mean exact-canonical realized Brier advantage at least `0.008`;
- its bootstrap interval is strictly above zero;
- simulated-to-realized advantage Spearman at least `0.25`;
- its bootstrap interval is strictly above zero;
- wins minus losses at least 15.

Positive advantage means the depth-three root has lower Brier.

The exact policy-level check must also show depth three versus myopic:

- at least 8% relative Brier reduction;
- a whole-tree Brier-difference interval strictly below zero;
- at least 40 wins among 64 trees.

Depth three versus depth two, fixed-support depth three, PTS, random root,
Hamming, coverage, and rank diagnostics are reported but cannot rescue or
veto the primary.

## Mechanics And Budget

- Exact accepted requests:
  `64 * (49 planning + 1 mechanics target + 8 validation) = 3,712`.
- HTTP attempts equal accepted requests plus retries.
- At most 24 total and provider-error retries.
- Zero reasoning tokens and forced exits.
- Every initial support has at least 16 valid rules, every validation support
  at least 16, every retained first branch at least 8, and every retained
  second branch at least 4.
- Hard confirmation cap: `$5.75`.
- Minimum authenticated starting balance: `$5.25`.

The result fails closed on schema, support, accounting, or budget errors.
There is no seed replacement, endpoint deletion, threshold change,
continuation, or efficacy rerun.
