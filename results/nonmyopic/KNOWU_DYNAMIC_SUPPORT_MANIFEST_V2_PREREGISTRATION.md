# KnowU Dynamic-Support Manifest V2 Preregistration

Frozen: 2026-07-26, after V1's representation-only failure and before any
profile-conditioned endpoint inspection or model call.

V2 preserves every source, task-family criterion, seed, split size, profile
world, and construction contract from
`KNOWU_DYNAMIC_SUPPORT_MANIFEST_PREREGISTRATION.md`.

The sole amendment is static goal parsing. A task's `GOAL_REQUEST` is accepted
when its AST recursively contains only:

- string constants;
- `+` concatenation of accepted expressions;
- f-string nodes containing string constants and **zero** formatted values.

Any runtime name, attribute, function call, subscript, or formatted value fails
the task. The gate still requires at least ten eligible families, all four
family splits nonempty, and at least three official profile worlds per selected
family.

Passing V2 authorizes only a separately preregistered two-family mechanics
run. No endpoint, profile value, or behavior-log content may be loaded by this
manifest. OpenRouter and OatML use remain zero.
