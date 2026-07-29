# Number Game DeepSeek Planner Exact-10 Serving Result

Date completed: 2026-07-29.

Status: **gated null; no formal run authorized**.

## Result

DeepSeek V4 Pro completed exactly 10 accepted requests in 10 HTTP attempts,
with zero retries, provider errors, reasoning tokens, or forced exits. Every
strict JSON response parsed. The two initial supports each contained 22 valid
unique executable rules. Total cost was `$0.0183607869`.

The frozen conditioned-support gate failed on one of eight cases:

| History depth | Valid unique counts |
|---|---|
| One observation | 14, 15, 18, 20 |
| Two observations | 7, 10, 13, 8 |

The `YES(10), NO(20)` two-observation response contained 24 schema-valid
rules, but two expressions were invalid and 15 collapsed to duplicate
extensions, leaving seven against the preregistered minimum of eight.

## Decision

The exact DeepSeek model, prompt, seed, thresholds, and response set are
closed. The support threshold is not reduced, the response is not repaired or
reissued, and the 2,336-call efficacy run is not launched. This is a serving
qualification null, not evidence about depth-three policy efficacy.

Artifacts:

- public `RESULT.json` SHA-256:
  `931c796a216fc1efa24a3b56650d9965110cd8241236e012f151f7aadbcfdc26`;
- private raw-response SHA-256:
  `b874699c809fbdf4178158ffd6b022e90258b4d6d1de75660ccc770701886478`.
