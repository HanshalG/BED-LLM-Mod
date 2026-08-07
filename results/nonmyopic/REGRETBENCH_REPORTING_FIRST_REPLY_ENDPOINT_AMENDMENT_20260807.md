# RegretBench Reporting Amendment: First-Reply Endpoint Alignment

Date: 2026-08-07

Status: frozen before any RegretBench policy response or endpoint.

## Dependencies

- original reporting protocol:
  `6594c91f2ceb5897ec0b2a0d1f6e58a5e4690325503414d9655aa45332be63f2`
- first-reply endpoint amendment:
  `cf5ae9d08ecfc4372611c58fcc1c1b32b67fd6830881c11ad0a352ba514df726`

## Primary Table Clarification

The mandatory primary policy table additionally reports, for every policy:

- truth-consistent exact first-reply likelihood-match rate; and
- likelihood-aligned two-action trajectory rate, which requires both the
  existing action-validity rule and a truth-consistent first-reply match.

The endpoint label and table caption state that action-invalid or
first-reply-unmodelled paths receive the frozen worst-case penalty. Raw masses
remain descriptive and are not substituted for scored values.

## Alignment-Complete Diagnostic

For each of `myopic_width`, `history_blind_depth2`, `fixed_depth2`, and
`random`, form the paired subset of tasks where both that control and
`dynamic_depth2` have a likelihood-aligned two-action trajectory. On each
subset, report:

- subset size;
- dynamic-minus-control Brier and log-loss mean, sample standard deviation,
  20,000-sample paired task-bootstrap 95% interval, and bootstrap probability
  of improvement;
- Brier wins, ties, and losses; and
- each policy's all-64 likelihood-aligned trajectory rate.

Bootstrap seeds are `202608151000 + 10*j` for Brier and the next integer for
log loss, where `j` follows the control order above.

The dynamic-versus-myopic subset earns the separately labelled
`alignment_complete_corroboration` flag only when all conditions hold:

1. at least 24 paired tasks;
2. mean dynamic-minus-myopic Brier at most `-0.01`;
3. bootstrap probability of Brier improvement at least `0.80`;
4. Brier wins exceed losses; and
5. mean dynamic-minus-myopic log loss is at most `0`.

This diagnostic cannot alter policy status, confirmation authorization, or a
frozen claim tier. If mechanics fail, it is not interpreted or reported as an
efficacy diagnostic. A positive full-sample result with failed alignment-
complete corroboration must be described as potentially involving differential
path-validity penalties rather than as penalty-independent evidence.

## Scope

This amendment changes no model call, prompt, support, candidate, selector,
trajectory, endpoint value, seed used by the experiment, cohort, science or
mechanics gate, confirmation rule, or budget. It only makes the new endpoint
validity rule visible and preregisters a non-rescuing robustness analysis.
