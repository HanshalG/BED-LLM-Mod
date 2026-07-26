# KnowU Dynamic-Support Mechanics V3 Preregistration

Date frozen: 2026-07-26

## Amendment

V2 serving failed because a synthetic clarification contained a conjunction.
V3 keeps the same parser and strengthens the prompt: every question must avoid
the literal words `and` and `or`, in addition to targeting one dimension.

V3 also fixes two instrumentation details without changing behavior:

- save every raw response batch before parsing so a failed parse remains
  auditable;
- set `reasoning_effort` to null rather than the accidental string `"none"`,
  making the adapter's `reasoning_enabled` flag false and sending no reasoning
  parameter.

The private fixture hash, model, temperature, support size, question count,
independent branch requests, semantic endpoint, threshold 70, exact 10/42
request counts, zero-retry rule, cost caps, and all scientific gates remain
unchanged from V1/V2.

If the V3 10-call serving gate fails, mechanics remains locked. If it passes,
run mechanics once without further prompt or parser changes.
