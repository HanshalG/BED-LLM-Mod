# tau-Knowledge Task Inventory Audit

Date: 2026-07-25

## Question

Can the existing held-out tau-Knowledge result be strengthened with a larger
genuinely untouched official banking-task confirmation?

## Source And Scope

- Official checkout:
  `sierra-research/tau2-bench@1d244f5dca42944b67a379b44bfeb9f5748f189d`
- Banking corpus: 698 documents and 97 tasks.
- Audit read task IDs and opening-format membership and compared them with
  frozen constants and public artifacts.
- It did not inspect required-document values or compute a new endpoint.
- No model calls were made.

## Accounting

All 97 tasks are already accounted for:

- 36 tasks matched the old narrow scripted-opening format and were used in the
  earlier zero-call development audit. This is why the first prospective
  retrieval script explicitly excluded `OLD_OPENING_PATTERNS`.
- Six additional tasks outside that 36-task set were among the separately
  inspected `PREVIOUSLY_AUDITED_IDS`.
- 28 fresh scripted-opening tasks became the V1 smoke, opportunity, and sealed
  first-link confirmation split.
- The remaining 27 unique tasks entered the V2 no-script pipeline. Together
  with three already inspected/reused mechanics tasks, V2's constants cover
  its two preprocessing checks, two smoke tasks, six opportunity tasks, and
  the final 20-task held-out receding-policy confirmation.

The union of those groups is exactly the 97 official task IDs. A repository-wide
search also confirms that the 32 IDs absent from current scorer constants are
the old-format development-audit cohort, not an untouched reserve.

## Decision

There is no honest new tau-Knowledge banking holdout to release. Additional
executions on the existing 20 tasks can measure provider or scorer stability,
as the completed retest and rank-ensemble blocks already do, but cannot add
task-level generalization or independent endpoint power.

Do not spend OpenRouter credit on a larger run labeled as fresh tau evidence.
The next external expansion requires a new task source or a prospectively
constructed benchmark.
