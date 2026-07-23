# Mushroom Feature Acquisition 26B Serving Smoke V4 Preregistration

Registered 2026-07-23 after S0 v3 failed its serving-format gate and before any
Mushroom policy score or proposal-quality endpoint was computed or inspected.

## V3 Failure

Job `106351` accepted seven cells, including the largest collected-state cell after
one corrected length retry. On cell 7, both responses had the correct four keys and
per-root lengths but used 11--12 lowercase alphabetic codes; even after uppercasing,
six--seven codes remained outside their branch-specific menu ranges. Any automatic
clipping or remapping would be ambiguous, so v3 failed closed. The run made ten
requests with 20,988 prompt tokens and 401 completion tokens, zero reasoning tokens,
zero forced exits, and zero API cost.

## Frozen Serialization Repair

- Use fresh smoke seed `24130` and a fresh output directory.
- Keep the same model, temperature zero, non-thinking mode, K4 machine-fixed roots,
  factored shared menus, branch probabilities, semantic labels, legal choices,
  128-token cap, one validation retry, scheduler, and all prior serving gates.
- Remove the artificial base-32 encoding. Each menu already displays a zero-based
  integer index, so return one integer per branch in four root-keyed arrays:
  `{"r0":[...],"r1":[...],"r2":[...],"r3":[...]}`.
- Require exact keys and array lengths. Reject booleans, non-integers, and indices
  outside each branch's displayed menu. Do not clip, truncate, remap, or fall back.

This changes only response serialization. Every accepted integer maps directly to
the same branch-specific action as its former code. V4 remains solely a serving and
mechanics gate; proposal quality remains quarantined.
