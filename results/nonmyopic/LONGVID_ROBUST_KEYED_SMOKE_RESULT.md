# LongVid Robust Keyed Serving Smoke Result

Date executed: 2026-07-29

Status: **all frozen transport and support-dynamics gates passed; the
preregistered two-pair mechanics run is authorized.**

## Integrity

- Preregistration commit: `a270530`
- Interface: `longvid-robust-keyed-belief-1`
- Model: `openai/gpt-5.4`, reasoning disabled, temperature `0`
- Public result SHA-256:
  `22e3dea8f34bca257c5550d8c47b361461666cecea95f69df354d013007b8de6`
- Private raw SHA-256:
  `ca41dd652864787a8e53464de4ef5a80b7c7ff6e66d169f6a6cae9463025d5bd`

The fixture is synthetic and contains no scientific LongVid task, caption, or
hidden evidence endpoint.

## Result

| Check | Result |
|---|---:|
| accepted requests | 10/10 |
| HTTP attempts | 10 |
| provider retries | 0 |
| parsed outputs | 10/10 |
| changed refreshes | 8/8 |
| unique refreshes | 8/8 |
| grounded anchors per refresh | 6, 6, 6, 6, 6, 6, 6, 5 |
| rank parsed | B, 77% |
| reasoning tokens / forced exits | 0 / 0 |
| prompt / completion tokens | 7,429 / 2,851 |
| cost | `$0.0613375` |

No semantic repair, completion reissue, parser extraction, or fallback was
used. The order-insensitive keyed codec therefore removes the earlier
nonsemantic row-order fragility without relaxing completeness or field
validation.

Per the frozen decision rule, execute the exact 22-call mechanics gate on rows
319 and 120 without changing model, prompts, parser, roots, thresholds, or
transport allowance.
