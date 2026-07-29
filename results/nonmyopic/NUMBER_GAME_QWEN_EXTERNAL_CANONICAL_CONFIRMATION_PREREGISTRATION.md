# Number Game Fresh Qwen External-Canonical Confirmation

Date frozen: 2026-07-29, after the 31-tree external replay and before any new
planning response or fresh-tree outcome.

## Question

Do wholly fresh Qwen-generated, path-dependent Number Game belief trees
produce a deeper-policy proper-score gain on the complete external canonical
target bank?

The prior 31-tree replay used already-open planning trees. It found depth three
below depth two by 5.70%, but its whole-tree interval narrowly crossed zero,
and it found a clean 13.79% gain over myopic EIG. This study is the single
fresh-tree confirmation of those two fixed comparisons.

## Frozen Design

- Planner: `qwen/qwen3.7-plus`, non-reasoning, temperature `0.7`.
- Independent validation generator: `google/gemini-2.5-flash`,
  non-reasoning, temperature `0.7`.
- 32 fresh planning seeds `49000..49031`.
- 32 generated-target seeds `49100..49131`; these targets are retained only
  for source-tree mechanics and are not used for root selection or efficacy.
- Eight independent validation supports per tree, seeds `49200..49455`.
- Retained-rejuvenation at both answer-conditioned refreshes.
- Cross-fitted depth-three and depth-two roots are selected only with the
  eight validation supports.
- Efficacy endpoint: all 33 Tenenbaum--Griffiths concepts, equal concept
  weight, with the already-frozen natural extension from `1..100` to
  `0..100`.
- Equal weight over 32 trees and the existing 20,000-draw whole-tree
  bootstrap.

The Qwen interface smoke is immutable:

- result SHA-256
  `2f311f9118a74f720edf219ca94e0ae4a8a01bd40b0ace01ade27e8d4c4795df`;
- 73/73 accepted HTTP calls, zero retries, provider retries, reasoning
  tokens, or forced exits;
- all support and branch mechanics gates passed.

This exceeds the required ten-call serving gate. No new smoke is purchased.

## Primary Gates

The result is `passed` only if all mechanics gates and all seven proper-score
primary gates hold:

- at least 12/32 depth-three and depth-two roots differ;
- depth three improves Brier over cross-fitted depth two by at least 1%, the
  whole-tree interval is strictly below zero, and depth three wins at least
  12 trees;
- depth three improves Brier over myopic EIG by at least 5%, the whole-tree
  interval is strictly below zero, and depth three wins at least 16 trees.

Hamming, exact truth-extension coverage, fixed-support depth three, PTS,
uniform random root, and external-bank rank fidelity are prespecified
diagnostics. They are all reported but cannot veto or rescue the Brier
primary. Any claim must state adverse diagnostics.

## Mechanics And Budget

- Exact accepted requests: `32 * (49 Qwen planning + 1 generated target +
  8 validation) = 1,856`.
- No endpoint-generation calls.
- Transport attempts must equal accepted requests plus bounded retries.
- At most eight transport retries; zero provider-error retries, reasoning
  tokens, and forced exits.
- Every initial support has at least 16 valid rules, every validation support
  at least 16, every retained first branch at least 8, and every retained
  second branch at least 4.
- Hard run cap `$4.00`; minimum authenticated starting balance `$3.60`.
- Expected spend from the prior exact interface: approximately `$2.8--3.5`.

The run fails closed on any schema, parse, budget, or support error. No
continuation, response repair, target deletion, family reweighting, seed
replacement, threshold change, or efficacy rerun is allowed.

## Interpretation

Passing both primary comparisons would provide fresh planner-family evidence
that non-myopic planning and the additional depth step improve calibrated
prediction on a literature-defined target distribution. A null leaves the
31-tree external result as directional supporting evidence only.
