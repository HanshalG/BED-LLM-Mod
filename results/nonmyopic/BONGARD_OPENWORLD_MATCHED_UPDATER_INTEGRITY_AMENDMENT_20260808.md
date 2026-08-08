# Bongard Matched-Updater Integrity Amendment

Frozen: 2026-08-08, before any Bongard mechanics, development, confirmation,
or scientific endpoint response.

The matched realized-updater policy introduced in the preceding amendment is
unchanged. This amendment makes its defining construction an explicit runtime
validity gate in every mechanics, development, and confirmation block.

For every task, the verifier must establish that
`history_blind_update_matched_first`:

- uses exactly the dynamic policy's first image, realized first label, first
  score, first score margin, and complete first-score map;
- has a second-score map over exactly the seven candidates remaining after the
  shared first query;
- selects the deterministic argmax of that second-score map; and
- records exactly the corresponding second-action selection margin.

The confirmation block previously inherited this construction from the shared
planner and independently replayed it, but did not name it in the block's own
validity conjunction. Confirmation now includes the same exactness gate already
used by development. Altering the matched first query, second-score support,
selected second query, or tie margin therefore makes the block fail closed.

This is a validity repair, not a scientific endpoint or a new estimand. It does
not change tasks, images, prompts, seeds, request ordering, policies, actions,
effect thresholds, endpoint definitions, sample sizes, request ceilings, or
budget ceilings. No model call or label access was used to select it.
