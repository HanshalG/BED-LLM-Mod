# RegretBench SMC Policy Reporting Protocol

Date frozen: 2026-08-07, before any RegretBench SMC policy model response.

Status: sealed development-reporting contract. It changes no policy, endpoint,
threshold, seed, cohort, model, request count, budget, or authorization.

## Scope And Verification

This protocol reports only the preregistered 64-task SMC dynamic depth-two
development policy. A report may be emitted only when the producer-independent
SMC verifier exactly reconstructs the raw banked parents, annotations, all
8,192 simulated transitions, frozen selections, realized transitions, privacy,
mechanics, scientific summaries, and public `RESULT.json`, with zero model
calls and zero mismatches.

The report generator reruns that verifier and requires byte-identical stored
verification before reading the result status. Result or verification tamper,
missing raw artifacts, partial execution, or a changed protocol produces no
report.

## Frozen Claim Tiers

Exactly one tier follows directly from the independently replayed development
status:

- `mechanics_failed` -> `smc_mechanics_failure_no_scientific_result`:
  no policy-efficacy result is reported and no comparison table is interpreted.
- `gated_null` -> `smc_development_policy_null_confirmation_forbidden`:
  the preregistered SMC development test is null and confirmation is forbidden.
- `passed` -> `smc_provisional_development_signal_confirmation_required`:
  SMC planning over retained/revised LLM semantic particles has a provisional
  development signal and independent confirmation is required.

No development result may be called confirmed. No pooled, subgroup, filtered,
alignment-complete, stable-draw, fresh-redraw, or optional-baseline analysis can
change these tiers.

The later prospectively frozen primary claim-gate amendment
`7f7140418bd08e207bf1f52e9838c93234acf66d5eec48cf9c9879302426df62`
controls `passed` versus `gated_null`. The report must show its 13 headline and
path-dependence gates, their conjunction, and the legacy conjunction of all 34
diagnostic booleans. All 34 values and all controls remain mandatory to report,
but only the 13 registered gates classify the SMC claim. A failed secondary
diagnostic cannot veto a passed primary conjunction, and no secondary result
can rescue a failed one.

## Mandatory Primary Report

The report contains all 64 tasks and all primary policies:

- `smc_dynamic_depth2`;
- `smc_myopic_refresh_brier` (headline horizon-isolating control);
- `smc_myopic_brier`;
- `smc_myopic_width`;
- `smc_history_blind_depth2`;
- `smc_fixed_depth2`; and
- `random`.

For every policy report mean, sample standard deviation, and task count for the
aligned generated-likelihood truth mass, Brier score, log loss, first-step
truth mass, supported/novel action indicators, exact second-reply matching,
posterior-parent update application, and valid two-action trajectory.

For dynamic versus every control report root disagreements, paired Brier and
log-loss mean differences, sample standard deviations, 95% bootstrap intervals,
improvement probabilities, and wins/ties/losses. Report the preregistered
predicted-to-realized Spearman diagnostics and every mechanics/science gate.

## SMC Mechanism And Non-Rescuing Diagnostics

The report states that initial hypotheses and questions are the exact banked
raw parent slots, while the LLM supplies aligned replies and all retained/revised
semantic transitions. It reports exact lineage/retention mechanics, posterior
parent-update rates, and request/cost accounting.

The existing two-draw stability diagnostic is reported as non-gating and
non-rescuing. An alignment-complete paired diagnostic may report only the subset
where both policies have valid aligned trajectories; it is descriptive and can
never alter status or tier. Fresh final SMC regeneration and the optional Luna
thinking baseline are separately labelled descriptive, unmatched endpoints and
cannot alter status or tier.

## Paper Mapping

The deterministic paper fragment must use the exact tier sentence, identify
`smc_myopic_refresh_brier` as the headline matched control, state that the LLM
owns semantic particle likelihoods and path-dependent retain/revise transitions,
and show no efficacy table after mechanics failure. It may replace the existing
conditional RegretBench generated fragment only; it does not alter the current
manuscript while no verified SMC result exists.

Freezing this protocol makes zero model calls and costs zero dollars.
