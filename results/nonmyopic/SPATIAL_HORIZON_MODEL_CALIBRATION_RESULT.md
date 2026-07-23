# Endpoint-Free Spatial Horizon Model Calibration

This model-selection calibration contained no Rock map, belief, EIG, posterior,
or scientific endpoint. Each of three rotated grid cases had a fixed first move,
one target reachable with two more moves plus inspection, and one distractor
requiring three moves before inspection.

| Model | Correct route content | Bare-JSON parser | Forced exits | Cost |
| --- | ---: | ---: | ---: | ---: |
| Qwen 3 32B dense thinking | `3/3` | `3/3` | `0` | `$0.00030824` |
| Gemma 4 31B dense thinking | `3/3` | `0/3` | `0` | `$0.00064646` |

Qwen returned exact compact JSON in every case. Gemma's three route arrays were
also exactly correct, but each was wrapped in a Markdown JSON fence; the
calibration parser intentionally accepted only a JSON object from character
zero. The h4 scientific compiler already accepts fenced JSON, so both models
show route-counting competence, while Qwen has the cleaner serving result.

Qwen 32B is selected for one separately preregistered h4 smoke. These generic
responses do not enter the scientific proposal pool or any endpoint.

Artifact: `spatial_horizon_model_calibration_20260723/CALIBRATION.json`.

## Frontier Ceiling Calibration

The selected Qwen 32B Rock smoke later failed closed and, as registered, ended
the open-weight model search. A separate endpoint-free ceiling calibration then
ran full GPT-5.4 with high reasoning on the same three generic cases:

| Model | Correct route and bare JSON | Forced exits | Reasoning tokens | Cost |
| --- | ---: | ---: | ---: | ---: |
| GPT-5.4 high reasoning | `3/3` | `0` | `456` | `$0.00936750` |

This authorizes one separately preregistered frontier h4 smoke on fresh Rock
cells. It does not alter or pool with any open-weight result.

Artifact: `spatial_horizon_gpt54_calibration_20260723/CALIBRATION.json`.
