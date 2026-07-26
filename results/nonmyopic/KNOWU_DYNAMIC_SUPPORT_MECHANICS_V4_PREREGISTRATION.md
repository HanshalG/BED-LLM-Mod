# KnowU Dynamic-Support Mechanics V4 Preregistration

Date frozen: 2026-07-26

## Amendment

V3 exposed an ASCII-only parser bug before any scientific endpoint:
semantically distinct Chinese strings all normalized to empty. V4 changes
only text normalization and continuation mechanics:

- normalize Unicode letters and numbers rather than ASCII `[a-z0-9]`;
- accept exactly one terminal ASCII `?` or full-width Chinese `？`;
- load, hash-check, and parse the exact six cached V3 initial responses;
- do not issue replacement initial-policy calls;
- run only the remaining 6 user-simulator, 24 isolated refresh, and 6
  post-policy judge calls.

Frozen continuation inputs:

- V3 private raw SHA-256:
  `3f9a1512b4514920a77086db0177c4d788915b3d3c4167e9ddb2181e7c47b11f`
- V3 public failure SHA-256:
  `2affc799cfd3378e8aee2d89c333e4d00bded80113eca7ddfd7160fe10a31281`
- cached requests/cost: 6 / $0.0403825

Composite accounting must be exactly 42 accepted requests and 42 HTTP
attempts, zero reasoning/retry/forced exit, and no more than $0.75. All
scientific gates, supports, questions, prompts, model, threshold, and fixture
remain unchanged.

This is the final authorized mechanics continuation. A parse, serving, budget,
or scientific failure closes this exact KnowU construction.
