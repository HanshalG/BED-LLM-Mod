# Strategy-Prior Successor Capability Probe

Registered on 2026-07-18 before the successor's first model call. This is distinct
from the closed original StrategyEIG protocol: its original L3 negative result remains
unchanged regardless of the outcome below.

## Question

The original L3 endpoint established a valid exact instrument but not LLM primacy. Its
only model-scale attempt, Gemma 4 31B with reasoning, failed before returning any final
strategy JSON. This successor asks the narrower capability question: can a stronger,
serving-verified reasoning model generate valid compact plans well enough to pass the
same paired L3 controls?

## Model and Serving Gate

The chosen generator is `qwen/qwen3.5-397b-a17b`, a current OpenRouter reasoning model
with a 262,144-token context window. Reasoning is enabled with a 512-token reasoning
budget and a separately reserved 256-token final-output budget. This avoids treating
the previous model's `None` final content as a policy result.

The mandatory first gate uses exactly ten no-repair generation cells: five strict Rock
L1 prompt/parser cells and five strict continuous L3 prompt/parser cells. The smoke
passes only if all ten first responses parse and execute under their respective strict
grammars, with no forced output exit. It has a `$0.05` hard cap. Any smoke failure is a
serving/interface result only; the capability endpoint is not launched and a different
model may be registered under the successor authorization.

## Formal Endpoint If and Only If the Gate Passes

The formal run is the frozen original continuous L3 design: fresh seed `31003`, 30
paired trials, 30 rounds, 64 particles plus truth, K=4, horizon 4, 64 CRN rollouts,
the same five arms, 10,000 paired-bootstrap replicates, and the exact legality,
shared-root, width-compute, grid-compute, and zero-rollout-LLM mechanics. The only
scientific difference is the strategy generator and its serving configuration.

The successor passes only if StrategyEIG has strictly positive paired 95% lower
entropy-gain endpoints against both grammar-matched random strategies and shared-d1
(and all original mechanics pass). Width and matched-budget grid-d2 remain required
controls and are reported alongside those gates.

## Cost Bound

The formal run retains the completed run's 1,373 physical-call reference volume. At
the configured maximum 768 completion tokens per call, Qwen's current catalog pricing
(`$0.39/M` prompt; `$2.34/M` completion) gives a conservative approximately `$2.67`
bound using the original 498,924 prompt-token count. The formal hard cap is `$3.00`;
the OpenRouter ledger independently enforces it. No horizon extension is authorized
unless this successor passes.

## Files

- Serving gate: `scripts/nonmyopic_strategy_successor_smoke.py`
- Smoke config: `configs/config_nonmyopic_strategy_successor_smoke_qwen397_thinking_openrouter.yaml`
- Formal config: `configs/config_nonmyopic_copex_strategy_l3_successor_qwen397_thinking_openrouter.yaml`
