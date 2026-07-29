# Number Game Grok Planner Exact-10 Serving Result

Date completed: 2026-07-29.

Status: **gated null; no formal run authorized**.

## Result

Grok 4.3 completed exactly 10 accepted requests in 10 HTTP attempts, with
zero retries, provider errors, reasoning tokens, or forced exits. Every
strict JSON response parsed. Initial supports contained 23 and 21 valid
unique executable rules. Total cost was `$0.0167385`.

The frozen conditioned-support gate failed:

| History depth | Valid unique counts |
|---|---|
| One observation | 8, 20, 12, 18 |
| Two observations | 0, 7, 15, 10 |

For the `YES(10), NO(20)` history, one expression was invalid and 23 of 24
schema-valid rules contradicted the stated history, leaving no usable
hypothesis. The opposite-label history retained only seven.

## Decision

The exact Grok model, prompt, seed, thresholds, and responses are closed. No
rule is repaired, response reissued, or threshold changed, and the 2,336-call
formal run is not launched. This is a serving qualification null, not an
efficacy result.

Artifacts:

- public `RESULT.json` SHA-256:
  `7200f75b093d089a6490c521e2b14c0afa2fdf92d7caad3fffa46d4708f59694`;
- private raw-response SHA-256:
  `001e8b880692b3faf9cfa7c254d16c83c241c5c8acfd4adedc3e7f9182e75300`.
