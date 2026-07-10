# Path E Step 0a: Real-Model Paprika Smoke

Status: **PASSED** on 2026-07-11 after automated and manual review.

## Canonical Run

- Run: `20260711T000402_paprika-step0a-openrouter-26b-nonthinking-v6`
- Commit: `8b84edf`
- Backend/model: OpenRouter, `google/gemma-4-26b-a4b-it`
- Configuration: five official Paprika customer-service eval tasks, two-round budget,
  one-step EIG, 12 initial hypotheses, five shared candidates, non-thinking questioner
  and answerer, seed 1304, concurrency 24.
- Cost: $0.04004851 for 642 requests, 135,065 prompt tokens and 63,832 completion
  tokens. Reasoning tokens and forced exits were both zero.

## Gate Evidence

- Five tasks and nine realized turns (one task resolved on its first turn).
- Clean answer mappings: 8/9 = 88.89%, above the preregistered 85% threshold.
- Structured retries: 1; terminal structured failures: 0; runtime failures: 0.
- Manual review found all eight accepted mappings semantically correct.
- The sole uncovered reply was terminal: the scale action proposed the exact released
  remedy (calibration with certified test weights), the native semantic success judge
  accepted it, and the customer answered "Please go ahead and try that" rather than an
  answer-space outcome. It did not feed a posterior update because the task stopped.
- The one resolved task was a genuine exact-remedy action; no false-positive diagnostic
  success remained.

The ignored raw artifacts remain under the canonical run directory, including
`paprika_step0_analysis.json` and `items/000_EIG/paprika_smoke.json`.

## Development Runs

Earlier runs are retained in the experiment ledger. They exposed and fixed, in order:
false diagnostic success, explicit observations mapped to uncertainty, leaky
post-action outcome sets, and explicit `null` likelihood mass. Cumulative OpenRouter
spend through the canonical smoke was $0.19704997 of the authorized $20 budget.
