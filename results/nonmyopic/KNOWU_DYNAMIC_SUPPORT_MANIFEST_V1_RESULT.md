# KnowU Dynamic-Support Manifest V1 Result

Date: 2026-07-26. Outcome: **zero-call representation gate failed**.

V1 required at least ten hard, multi-profile, interactive task families with a
literal nonempty `GOAL_REQUEST`. Nine passed. The split itself remained
nonempty and all selected families had at least three official profile worlds.

The tenth source-audited family,
`SearchTopInfoPreferenceAskUserTask`, was excluded because its constant goal is
represented as a parenthesized concatenation containing an `f"..."` AST node.
Inspection after the V1 failure showed that this f-string contains no
interpolation or profile-dependent value. V1 remains failed and is not
rewritten.

A V2 format-only amendment is permitted before endpoint inspection. It may
accept a `GOAL_REQUEST` only when static AST evaluation proves that every
component is a constant string or a zero-slot f-string. Runtime interpolation,
attribute access, calls, names, and profile-dependent expressions remain
invalid. Seed, split sizes, task criteria, profiles, endpoints, and method are
unchanged.

OpenRouter calls: `0`. OpenRouter spend: `$0`. OatML jobs: `0`.
