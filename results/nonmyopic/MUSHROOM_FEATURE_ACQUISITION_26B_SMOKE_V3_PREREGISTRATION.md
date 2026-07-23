# Mushroom Feature Acquisition 26B Serving Smoke V3 Preregistration

Registered 2026-07-23 after S0 v2 failed its serving-format gate and before any
Mushroom policy score or proposal-quality endpoint was computed or inspected.

## V2 Failure

Job `106348` accepted all six uncollected cells on their first attempt. On the first
collected-state cell, both bounded attempts returned a syntactically valid string of
62 menu codes instead of the required 35. The run therefore failed closed after six
accepted cells. It made eight requests with 41,867 prompt tokens and 218 completion
tokens, zero reasoning tokens, zero forced exits, and zero API cost.

## Frozen Serialization Repair

- Use fresh smoke seed `24129` and a fresh output directory.
- Keep the same model, temperature zero, non-thinking mode, K4 machine-fixed roots,
  posterior branch probabilities, semantic labels, legal branch choices, base-32
  codes, 128-token cap, one validation retry, scheduler, and all v1/v2 gates.
- In this deterministic feature-acquisition environment, every outcome branch under
  a fixed root has the same legal follow-up menu. Show that shared menu once per root
  instead of repeating it inside every branch.
- Return one separately length-checked code string per root:
  `{"r0":"...","r1":"...","r2":"...","r3":"..."}`. Within each root, codes
  remain ordered by the listed outcome branches and choose independent follow-ups.

This changes only prompt/response serialization. It does not alter any candidate
policy, posterior, branch probability, action, or score. V3 remains solely a serving
and mechanics gate; proposal quality remains quarantined.
