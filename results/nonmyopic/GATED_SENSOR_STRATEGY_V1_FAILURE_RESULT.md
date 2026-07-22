# Gated Sensor Strategy v1: Failed-Closed Interface Result

## Decision

The preregistered v1 LLM confirmation produced no policy endpoint. It failed closed because Gemma 4 26B A4B, at temperature zero, could not reliably copy globally legal action strings into branch-specific menus. Repeated identical-config resumes eventually plateaued on deterministic invalid responses.

## Audit

| Item | Value |
| --- | ---: |
| Accepted and revalidated cells | 385 |
| Rejected attempts | 662 |
| Trials represented in accepted cache | 30/30 |
| Accepted h2 cells | 345 |
| Accepted h1 cells | 40 |
| Registered terminal repairs | 27 |
| Physical requests | 1,047 |
| Total cost | `$0.76078769` |

| Rejection class | Count |
| --- | ---: |
| Illegal follow-up action | 378 |
| Illegal root action | 156 |
| Wrong outcome keys | 115 |
| Invalid JSON | 11 |
| Incomplete JSON fence | 2 |

## Interpretation

The exact verifier did its job: no invalid branch entered scoring or execution. The failure localizes to the proposal representation. The prompt required the model to reason about posterior uncertainty while also copying action IDs whose legality changed with the active panel and branch. At temperature zero, resume could not escape cells where the same illegal cross-panel precise action was reproduced.

This does not estimate StrategyEIG performance. The next interface should remove global string copying from the model's responsibility: machine-assign legal root slots and expose branch-local numbered follow-up choices. A fresh smoke and preregistration are required before evaluating that interface.

## Artifact

The raw failed-closed cache, every accepted response, every rejected response, and the final run usage are preserved in `results/nonmyopic/gated_sensor_strategy_confirmation_20260722/FAILURE.json`.
