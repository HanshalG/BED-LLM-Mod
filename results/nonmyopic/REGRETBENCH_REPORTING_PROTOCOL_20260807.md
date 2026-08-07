# RegretBench Frozen Reporting Protocol

Date frozen: 2026-08-07

## Purpose

Fix the RegretBench tables, claim tiers, and null/failure language before any
model response. The reporting layer cannot rescue or reclassify a preregistered
result and cannot merge the primary aligned endpoint with secondary fresh
regeneration.

## Admissible Inputs

A report requires a complete `RESULT.json` and its independently generated
`VERIFICATION.json`. The verification must have status `verified`, zero model
calls and cost, no mismatches, and an artifact hash for `RESULT.json` equal to
the file on disk. An unverified or altered result produces no report.

Development and confirmation are always reported separately. A pooled
128-task analysis may later be descriptive, but cannot change either stage's
status or claim tier.

## Frozen Claim Tiers

Exactly one tier is emitted:

1. `mechanics_failure_no_scientific_result`: a verified result has status
   `mechanics_failed`; no efficacy estimate is interpreted.
2. `development_policy_null_confirmation_forbidden`: development has status
   `gated_null`; confirmation remains closed.
3. `provisional_development_signal_confirmation_required`: development has
   status `passed`; this is positive development evidence only.
4. `confirmation_null_development_not_confirmed`: an authorized confirmation
   has status `gated_null`; the development signal did not independently
   confirm.
5. `confirmed_llm_native_nonmyopic_signal`: confirmation has status `passed`
   and independent replay verifies it.

No optional Luna result, secondary endpoint, random comparison, pooled result,
subgroup, or favorable individual gate can change the tier. A verification
failure has no tier because no report is generated.

## Mandatory Primary Table

For each of `dynamic_depth2`, `myopic_width`, `history_blind_depth2`,
`fixed_depth2`, and `random`, report mean and sample standard deviation across
all 64 tasks for:

- aligned terminal truth mass, Brier, and log loss;
- truth mass after question one;
- valid two-action trajectory rate;
- supported first-action rate;
- supported second-action rate;
- novel second-action rate; and
- exact generated-likelihood reply-match rate.

The table title must say that Brier and log loss use the aligned generated
likelihood endpoint and that invalid trajectories receive the frozen penalty.

## Mandatory Paired Comparisons

For dynamic versus myopic, history-blind, fixed, and random controls, report:

- dynamic-minus-control Brier mean, sample SD, 95% paired task-bootstrap
  interval, and bootstrap probability of improvement;
- wins, ties, and losses;
- dynamic-minus-control log-loss mean, sample SD, and 95% interval; and
- first-root disagreement count.

Also report the frozen predicted-to-realized dynamic-versus-myopic Spearman,
its interval, probability positive, and changed-root sample size. Every
individual science gate and its conjunction remain visible.

## Secondary And Optional Evidence

Fresh-regeneration comparisons are shown in a separate table labelled
`secondary_descriptive`; they never replace the aligned endpoint. If the Luna
naive-thinking baseline is available in development, show it in a separate
`optional_unmatched_compute_descriptive` section using only its fresh endpoint.
If unavailable, report its exact banked status. Confirmation has no Luna
baseline.

## Mechanics, Usage, And Wording

The report includes every mechanics gate, CRN counts, accepted requests, HTTP
attempts, retries, reasoning tokens, forced exits, and cost. It must state the
task count, model, reasoning mode, endpoint label, split, and independent
verification hash.

Frozen one-sentence interpretations are:

- mechanics failure: "The frozen mechanics contract failed, so no RegretBench
  policy-efficacy result is available.";
- development null: "The preregistered RegretBench development policy test was
  null; confirmation is forbidden.";
- development pass: "RegretBench shows a provisional development signal for
  LLM-native non-myopic planning; independent confirmation is required.";
- confirmation null: "The RegretBench development signal did not confirm on
  the untouched confirmation cohort.";
- confirmation pass: "RegretBench independently confirms a non-myopic gain
  over the LLM's own path-dependent semantic belief dynamics.".

The report generator writes only JSON and Markdown under the result directory.
It does not edit the manuscript or claim manifest automatically. Those files
may be updated only from the generated report and its exact hashes.

This protocol makes zero model calls and costs `$0`.
