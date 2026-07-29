# Number Game Fresh Qwen External-Canonical Replication V2

Date frozen: 2026-07-29, after the first fresh Qwen confirmation and before
any response from seeds `55000..55455`.

## Question

On wholly fresh LLM-generated belief trees, does path-dependent depth-three
planning improve proper-score performance over myopic EIG on the complete
external canonical Number Game concept bank?

The first Qwen cohort showed an 11.37% Brier reduction over myopic EIG,
whole-tree interval `[-0.01904,-0.00784]`, and 25/32 tree wins. It also
directionally passed every depth-three-versus-depth-two efficacy threshold,
but its formal status was null because two transient provider retries
violated a frozen zero-provider-retry mechanics gate. V2 is an independent
fresh-seed replication with transport retries bounded and reported rather
than interpreted as scientific failure.

## Frozen Design

- Planner: `qwen/qwen3.7-plus`, non-reasoning, temperature `0.7`.
- Validation and mechanics target generator:
  `google/gemini-2.5-flash`, non-reasoning, temperature `0.7`.
- 32 fresh tree seeds `55000..55031`.
- 32 mechanics-only target seeds `55100..55131`.
- Eight validation supports per tree, seeds `55200..55455`.
- Retained-rejuvenation at both answer-conditioned belief refreshes.
- Cross-fitted depth-three roots are selected only with the eight validation
  supports.
- Efficacy endpoint: all 33 Tenenbaum--Griffiths concepts, equal concept and
  tree weight, exact extensions over `0..100`.
- Existing 20,000-draw whole-tree bootstrap.
- Generated mechanics targets never enter efficacy or root selection.

The immutable 73-call Qwen serving smoke remains hash-bound at
`2f311f9118a74f720edf219ca94e0ae4a8a01bd40b0ace01ade27e8d4c4795df`.
It had zero retries, provider errors, reasoning tokens, and forced exits, and
all support gates passed.

## Primary

The result passes only if all mechanics gates hold and depth three:

- improves mean Brier over myopic EIG by at least 8%;
- has a whole-tree bootstrap interval strictly below zero;
- wins at least 20 of 32 trees.

Depth three versus cross-fitted depth two is prespecified as a diagnostic,
not part of the primary monotonic-depth claim. Fixed-support depth three,
positive-test strategy, uniform random root, Hamming, exact coverage, and
rank fidelity are also reported diagnostics and cannot rescue or veto the
primary.

## Mechanics And Budget

- Exact accepted requests:
  `32 * (49 planning + 1 mechanics target + 8 validation) = 1,856`.
- HTTP attempts must equal accepted requests plus retries.
- At most eight total retries and at most eight provider-error retries.
- Zero reasoning tokens and forced exits.
- Every initial support has at least 16 valid rules, every validation support
  at least 16, every retained first branch at least 8, and every retained
  second branch at least 4.
- Hard run cap `$3.50`; minimum authenticated starting balance `$3.25`.

The run fails closed on schema, parse, budget, support, or accounting errors.
No continuation, response repair, target deletion, family reweighting, seed
replacement, threshold change, or efficacy rerun is allowed.
