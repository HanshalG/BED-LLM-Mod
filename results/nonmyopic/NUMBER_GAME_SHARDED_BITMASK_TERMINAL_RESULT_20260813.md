# Number Game Sharded-Bitmask Terminal Result

The exact pushed sharded interface is terminally closed. All 30 Qwen proposal shards
returned parseable strict JSON with clean `stop` finishes under the 1,300-token cap,
zero retries, and zero reasoning tokens. Sharding therefore solved the measured
response-length failure.

The value-free history-obedience parser then failed before any Luna audit: every one of
the 30 shards contained at least one hypothesis whose 101-bit mask contradicted an
observed answer. Across all 240 generated hypotheses, 163 (`67.92%`) had at least one
observed-bit mismatch. Every aggregate draw was affected; bad-item counts ranged from
12 to 23 of 24.

The run made 30 Qwen calls costing `$0.04743296`, zero Luna calls, and opened no Number
Game target, source tree, policy endpoint, or scientific outcome. Fully posted account
usage is `$220.297957126`, making the Aug 13 account-wide spend `$0.163828246` from the
frozen `$220.134128880` boundary.

This is a serving/mechanics null, not evidence about non-myopic planning. It closes the
exact sharded-mask prompts, histories, seeds, and parser. The evidence supports a
prospectively distinct factorized interface in which Qwen generates semantic rule
descriptions, one history-blind DeepSeek call derives extensions, and a separate
history-blind DeepSeek call audits probe memberships without seeing those extensions.
It does not support repairing masks, overriding observed bits, or reusing these
descriptions.
