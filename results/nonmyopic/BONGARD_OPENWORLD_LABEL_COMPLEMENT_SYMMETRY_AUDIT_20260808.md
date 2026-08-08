# Bongard OpenWorld Label-Complement Symmetry Audit

Date: 2026-08-08 (Europe/London)

Status: **pass; zero model calls and zero scientific endpoint access**.

This audit was fixed and run before any Bongard mechanics, development, or
confirmation response. It changes no model, prompt, task, seed, action,
endpoint, request count, gate, threshold, or budget.

## Adversary

The test applies one global transformation to a complete synthetic planning
instance:

- every observed, simulated, realized, and endpoint label is negated;
- every particle likelihood `P(positive | rule, image)` is replaced by
  `1 - P(positive | rule, image)`;
- dynamic branch keys `(query, label)` are relabelled to
  `(query, not label)`;
- history-blind branches, particle weights, image IDs, endpoint IDs, and
  random-policy seeds are otherwise unchanged.

This is a strict class-name symmetry. A correct binary BED implementation may
change the displayed positive/negative names, but it must not change query
utilities, selected images, or proper endpoint scores.

## Executable Checks

The core test verifies numerical equality to absolute tolerance `1e-12` for:

- endpoint-predictive myopic EIG;
- fixed-support depth-two utility;
- answer-conditioned dynamic-support depth-two utility;
- history-blind depth-two utility after the analytical first-label update;
- endpoint truth probability, Brier score, log loss, and accuracy.

The policy-level test additionally verifies all registered mechanics policies:

- `myopic_width`;
- `fixed_depth2`;
- `fixed_score_dynamic_update`;
- `dynamic_depth2`;
- `history_blind_update_matched_first`;
- `shuffled_dynamic_depth2`;
- `history_blind_depth2`;
- `random`.

For every policy, the first and second image selections are unchanged, both
reported labels are complemented, the complete final history is complemented,
and every non-random second-step score map is numerically unchanged.

## Result

The focused core and mechanics suite passed `25/25`, and the complete Bongard
regression family passed `176/176`. The adversary found no class-indexing
asymmetry, label-sign leakage, or positive-class advantage in the planner or
endpoint scorer. The frozen August 10 paid chain remains unchanged; this audit
provides additional validity evidence only and cannot authorize a call, rerun,
development stage, or scientific claim.
