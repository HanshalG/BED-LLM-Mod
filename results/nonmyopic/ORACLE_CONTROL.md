# Restricted-Pool Oracle Control

## Scope

This is an exact scripted 20 Questions control with no LLM calls. Animal identity is the target, the answerer is the frozen UCI Zoo trait matrix, and the endpoint is deterministic MAP identity decoding. The preregistration is `ORACLE_CONTROL_PREREGISTRATION.md`.

## Exhaustive Pool

| Candidate pool | d1 AUC | d2 AUC | d2 - d1 AUC | 95% CI | Round accuracy delta |
| --- | ---: | ---: | ---: | --- | --- |
| all traits | 0.2834 | 0.2344 | -0.0490 | [-0.0553, -0.0429] | +0.0000, +0.0000, +0.0000, -0.0010, -0.0175, -0.1575, -0.2160, +0.0000 |

## Restriction and Width

| K | d1 AUC | d2 AUC | d2 - d1 AUC | 95% CI | first-action disagreement | d1 width control(s) |
| ---: | ---: | ---: | ---: | --- | ---: | --- |
| 2 | 0.1537 | 0.1684 | +0.0147 | [+0.0070, +0.0222] | 0.186 | d2-d1(K=4) -0.0406; d2-d1(K=8) -0.0900 |
| 3 | 0.1866 | 0.2024 | +0.0158 | [+0.0078, +0.0239] | 0.235 | d2-d1(K=6) -0.0386; d2-d1(K=12) -0.0730 |
| 4 | 0.2090 | 0.2165 | +0.0075 | [-0.0005, +0.0155] | 0.272 | d2-d1(K=8) -0.0419; d2-d1(K=16) -0.0673 |
| 6 | 0.2411 | 0.2314 | -0.0097 | [-0.0172, -0.0021] | 0.330 | d2-d1(K=12) -0.0441; d2-d1(K=21) -0.0520 |
| 8 | 0.2584 | 0.2369 | -0.0216 | [-0.0282, -0.0150] | 0.327 | d2-d1(K=16) -0.0469; d2-d1(K=21) -0.0465 |

## Noise Frontier

| Noise SD (nats) | d2 - d1 AUC | 95% CI | First-action disagreement |
| ---: | ---: | --- | ---: |
| 0.000 | +0.0158 | [+0.0078, +0.0239] | 0.235 |
| 0.025 | +0.0135 | [+0.0051, +0.0219] | 0.292 |
| 0.050 | +0.0188 | [+0.0103, +0.0273] | 0.344 |
| 0.100 | +0.0121 | [+0.0034, +0.0209] | 0.440 |
| 0.200 | +0.0088 | [-0.0010, +0.0184] | 0.559 |
| 0.400 | +0.0054 | [-0.0050, +0.0160] | 0.627 |

## Decision

- Restricted gap with a paired 95% CI excluding zero: `True`.
- Primary K=3 depth gain: `+0.0158` at zero noise and `+0.0054` at the largest injected noise.
- Noise acts as the preregistered headwind: `True`.
- First noise level with a CI including zero: `0.2`.
- **Verdict: `proceed_to_llm_exploration`.**

## Reproduction

```bash
python scripts/nonmyopic_oracle_control.py
pytest -q tests/test_nonmyopic_oracle_control.py
```
