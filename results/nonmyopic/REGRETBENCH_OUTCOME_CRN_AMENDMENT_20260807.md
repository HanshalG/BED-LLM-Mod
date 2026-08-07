# RegretBench Outcome-Level CRN Amendment

Date: 2026-08-07

Status: frozen before any RegretBench support or policy model response.

## Problem

The root-CRN amendment verifies requested seed formulas, but a provider may
ignore a seed or implement nondeterministic serving. Requested equality alone
does not prove that root comparisons share model randomness.

## Observable Check

For each development `(task, simulated_hypothesis, draw)` tuple, the four
history-blind requests use the exact same prompt and exact same requested seed.
Their strict parsed supports must therefore be identical when the seeded model
contract is honored. There are exactly:

```text
64 tasks * 8 hypotheses * 2 draws = 1,024 groups
```

## Amendment

- Canonically hash each parsed history-blind support.
- Require exactly four records and one support hash in every one of the `1,024`
  groups.
- Publish group count, exact group count, and exact fraction under a CRN
  diagnostic; raw supports remain private.
- Make `all_blind_crn_replays_exact` a primary mechanics gate.
- Independently reconstruct the diagnostic and gate from `RAW_BRANCHES.json`.

Conditioned prompts differ by root, so their token-level random streams cannot
be directly observed. Exact blind replay is the frozen empirical check that the
advertised seed is actually deterministic on this run. Failure banks a
mechanics failure and opens no scientific endpoint.

No prompt, task, seed, request count, score, scientific threshold, or budget
changes. This amendment cannot improve a result.
