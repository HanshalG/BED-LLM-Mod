# Rock L1 Gemma 31B Thinking Probe: Failed Closed

The one authorized model-scale probe did not produce an L1 policy endpoint and must
not be interpreted as a capability comparison.

## Frozen Probe

- Run ID: `nonmyopic-rock-strategy-l1-gemma31b-thinking-probe-20260716`.
- Generator: `google/gemma-4-31b-it`, OpenRouter reasoning enabled, 1,024 reasoning
  tokens plus 256 final-output tokens.
- Seed 12034; both Rock maps; 30 paired trajectories/map; 8 rounds; K=4; horizon 2.
- Exact Rock dynamics/scoring, all five arms, grammar, one validation retry, and
  `$2.00` aggregate hard cap were unchanged.

## Outcome

The persistent recovery wrote `L1_FAILURE.json` with:

- `status: failed_closed` and `error: strategy cell failed after 2 attempts:
  strategy response is not valid JSON`;
- zero accepted strategy or width cells, zero policy traces, and zero gate values;
- 60 initial strategy-cell rejects and 60 registered feedback retries, all with the
  same parse error; and
- 120 forced thinking exits in the contemporaneous usage snapshot.

Every recorded rejected response had final content `None`: the model consumed its
configured token budget in reasoning and emitted no parseable final JSON. This is an
interface/response-budget failure, not evidence that a stronger model loses to random
strategies or shared d1.

The concurrent runner had already submitted additional calls when the first fail-closed
exception was raised. It was stopped immediately after the failure artifact existed to
avoid spending toward an unavailable endpoint. The final shared ledger records 185
requests, 186,893 prompt tokens, 236,797 completion tokens, 90,263 reasoning tokens,
and `$0.114104675` for the one aggregate run ID. The artifact's own 126-request usage
snapshot is earlier because it was written before those already in-flight calls settled.

## Decision

The registered execution recovery was the only technical recovery. No larger token
budget, prompt change, model substitution, fresh seed, or further L1/L2/L3 run is
authorized. The strategy-prior conclusion therefore remains: the completed 26B L3
study did not establish LLM-primary non-myopic BED; the one permitted stronger-model
probe was unavailable due fail-closed response-interface behavior.

Artifacts:

- `results/nonmyopic/rock_strategy_l1_gemma31b_thinking_probe/20260716/L1_FAILURE.json`
- `results/nonmyopic/ROCK_STRATEGY_L1_MODEL_SCALE_PROBE_PREREGISTRATION.md`
- `results/nonmyopic/ROCK_STRATEGY_L1_MODEL_SCALE_PROBE_EXECUTION_RECOVERY.md`
