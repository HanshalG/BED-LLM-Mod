# KnowU Dynamic-Support Manifest Preregistration

Frozen: 2026-07-26, before the task-family split and before reading any
profile-conditioned task endpoint.

## Source

Use only KnowU-Bench commit
`c03a825991ede13add6631f2ed19b90755930dc6` and the preference-task, profile,
and clean-log hashes pinned in `KNOWU_BENCH_SOURCE_AUDIT_RESULT.md`.

## Eligible Families

A task family is eligible exactly when its released class:

- is tagged `hard`;
- is tagged `agent-user-interaction`;
- has a static nonempty `GOAL_REQUEST`;
- supports at least three official released profiles.

The zero-call gate requires at least 10 eligible families.

## Split

Sort eligible class names, shuffle with Python `random.Random(24415)`, then
assign:

- mechanics: first 2 families;
- opportunity: next 3;
- development: next 2;
- holdout: all remaining families.

All splits must be nonempty, and every selected family must retain at least
three official profile worlds. Families, not profile instances, are split so no
task semantics cross from development into holdout.

## Frozen Construction Contract

- Hidden world: one supported official profile, sampled uniformly.
- Policy input: task goal plus task-relevant clean behavioral logs; never a
  profile label or full profile YAML.
- Question action: one atomic task-preference question. Questions requesting a
  profile label, listing profile alternatives, bundling several independent
  preferences, or directly requesting the complete terminal action are invalid.
- Belief state: hypotheses generated in natural language by the LLM and
  regenerated from the realized history after each answer.
- Simulator: profile-conditioned KnowU user prompt with full dialogue history.
- Primary endpoints: truth-support recall, truth log probability, entropy AUC,
  and a task-conditioned terminal-preference score.
- Required controls: dynamic depth two, dynamic myopic, matched-compute myopic,
  and random atomic question selection under paired profile worlds.

Passing this manifest authorizes only a separately preregistered, fixture-clean,
cost-capped two-family mechanics run. It does not authorize an efficacy claim.
No OatML work is permitted.
