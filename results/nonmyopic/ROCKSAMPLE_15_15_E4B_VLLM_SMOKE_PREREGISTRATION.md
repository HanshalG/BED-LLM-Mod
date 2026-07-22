# RockSample[15,15] Gemma 4 E4B Serving Gate

Registered 2026-07-22 before any Gemma 4 E4B response on the 15-rock prompt.

## Purpose

Test whether the confirmed 15-rock root-slot policy can use a substantially smaller
non-thinking proposer without changing the exact verifier. This is an engineering
and proposal-mechanics gate, not a policy-effectiveness experiment. No paired
trajectory endpoint will be inspected in this stage.

## Frozen Design

- Frozen POBAX `15-15` geometry, start `(0,7)`, exact 32,768-state belief,
  diagnosis-only actions, and half-efficiency distance `log(2)`.
- Hugging Face checkpoint `google/gemma-4-E4B-it`, direct vLLM on one A100,
  bfloat16, non-thinking, temperature zero, maximum context 32,768, and maximum
  completion 2,048 tokens.
- The existing ten deterministic actual-prompt probe states, K4,
  `branch_policy_v2`, eight h2 cells and two h1 cells, concurrency one.
- Run on `msc,llm`, exclude `oat12`, and request one A100. This follows the active
  cluster constraint; no GH200 allocation is requested.

The smoke passes only if all ten cells parse within the registered one-repair bound,
all accepted strategies are legal, every h2 cell contains at least one movement root
and one direct-check root, and every h2 cell contains a movement root whose `none`
continuation is a check. Any terminal cell failure fails the smoke. Exact EIG and the
best-proposed/exhaustive-d2 fraction are descriptive diagnostics and cannot rescue or
fail the registered mechanics gate.

If the smoke passes, a fresh-seed 30-pair formal protocol and its statistical gates
will be written and committed before any formal model response. If it fails, no E4B
formal run will be launched under this protocol.

## Frozen Command

```bash
sbatch --job-name=r15-e4b-smoke scripts/run_nonmyopic_rock_strategy_a100.sh \
  scripts/nonmyopic_rock_branch_strategy_smoke.py \
  --config configs/config_nonmyopic_rocksample_15_15_e4b_vllm.yaml \
  --maps 15-15 --probe-states-per-map 10 --num-strategies 4 --concurrency 1 \
  --run-id nonmyopic-rocksample-15-15-e4b-vllm-smoke-20260722 \
  --output-dir results/nonmyopic/rocksample_15_15_e4b_vllm_smoke_20260722
```

## Execution Status

Pending.
