# RockSample[15,15] Gemma 4 12B Proposal-Quality Gate

Registered 2026-07-22 after the E4B transfer endpoint was fixed and before any
Gemma 4 12B response on the 15-rock prompt.

## Purpose

Test whether a dense middle-capacity, non-thinking proposer supplies enough valuable
K4 branches to justify a formal 15-rock transfer run. This gate responds to the
clean E4B failure: mechanics alone were insufficient because E4B covered only 9.6%
of exhaustive d2 value in the smoke and 6.8% in the formal run. The successful 26B
smoke covered effectively 100% on the identical fixed probes and 41.1--47.6% across
formal trajectories.

## Frozen Design

- Frozen POBAX `15-15` geometry, start `(0,7)`, exact 32,768-state belief,
  diagnosis-only actions, and half-efficiency distance `log(2)`.
- Hugging Face checkpoint `google/gemma-4-12B-it`, direct vLLM on one A100,
  bfloat16, non-thinking, temperature zero, maximum context 32,768, and maximum
  completion 2,048 tokens.
- The existing ten deterministic actual-prompt probe states, K4,
  `branch_policy_v2`, eight h2 cells and two h1 cells, concurrency one.
- Run on `msc,llm`, exclude `oat12`, and request one A100.

## Frozen Gate

All conditions are required:

1. All ten cells parse within the registered one-repair bound, all strategies are
   legal, and there are zero terminal cell failures.
2. Every h2 cell contains at least one movement root and one direct-check root, and
   contains a movement root whose `none` continuation is a check.
3. Across the eight h2 cells, mean best-proposed/exhaustive-d2 value is at least
   `0.40`.
4. At least six of the eight h2 cells have best-proposed/exhaustive-d2 value at
   least `0.20`.

These numerical thresholds were fixed from the already observed 26B/E4B separation,
not from any 12B response. If all conditions pass, a fresh-seed 30-pair formal
protocol will be written and committed before formal responses. If any condition
fails, no 12B formal run follows under this protocol. There is one smoke attempt;
model-load or runtime repair before any response may fix serving compatibility but
cannot change prompts, thresholds, K, or probe states.

## Frozen Command

```bash
sbatch --job-name=r15-12b-smoke scripts/run_nonmyopic_rock_strategy_a100_singularity.sh \
  scripts/nonmyopic_rock_branch_strategy_smoke.py \
  --config configs/config_nonmyopic_rocksample_15_15_12b_vllm.yaml \
  --maps 15-15 --probe-states-per-map 10 --num-strategies 4 --concurrency 1 \
  --min-mean-best-exhaustive-fraction 0.40 \
  --min-cell-best-exhaustive-fraction 0.20 --min-cells-at-or-above 6 \
  --run-id nonmyopic-rocksample-15-15-12b-vllm-smoke-20260722 \
  --output-dir results/nonmyopic/rocksample_15_15_12b_vllm_smoke_20260722
```

## Execution Status

The initial job `106170` reached `msc` node `oat16` but failed before model
initialization or any LLM response. The node-local vLLM build predates
`Gemma4UnifiedForConditionalGeneration`; upgrading Transformers alone cannot add the
missing vLLM architecture. Per the frozen serving-repair allowance, the identical
smoke will rerun through the repository's established Singularity pattern using the
official `vllm/vllm-openai:gemma4` image, which contains Gemma 4 Unified support.
The checkpoint, prompts, K, probes, thresholds, and one-repair policy are unchanged.

Job `106187` also failed before model initialization or any response because the
node's cached mutable `gemma4` image resolved to vLLM 0.19, while Unified support was
added in vLLM 0.23.0. The next serving-only repair pins
`vllm/vllm-openai:v0.23.0` and uses a fresh dependency directory containing only
PyYAML, pomdp-py, and openai-harmony without overriding the container's numerical,
Transformers, or pydantic stack. The scientific protocol remains unchanged.

Pinned-image job `106206` successfully built and cached the v0.23.0 SIF on `oat14`,
then failed in the repository preflight before model creation or any response because
pomdp-py's Gaussian module imports SciPy. The isolated dependency list now adds pinned
`scipy==1.17.1` without dependencies; the container's numerical stack and every
scientific setting remain unchanged.

Job `106215` then resolved `Gemma4UnifiedForConditionalGeneration` under vLLM 0.23.0
but failed at CUDA initialization before loading weights or producing a response: the
default v0.23.0 image targets CUDA 13 while `oat14` exposes driver compatibility 12.5.
Docker Hub publishes the official pinned `v0.23.0-cu129` image. The next serving-only
repair uses that image with vLLM CUDA compatibility enabled on A100 and a fresh
versioned dependency path. The model and all registered scientific settings remain
unchanged.

Job `106220` completed on `msc` node `oat14` with the pinned
`vllm/vllm-openai:v0.23.0-cu129` image and passed every frozen gate. All ten cells
parsed without repair or terminal failure; every h2 cell contained movement and
direct-check roots plus a move-then-check continuation. Mean best-proposed/exhaustive
d2 value across the eight h2 probes was `0.4217`, exceeding the registered `0.40`
threshold, and seven of eight probes exceeded `0.20`, exceeding the registered six.
The run used ten requests, 41,976 prompt tokens, 3,120 completion tokens, zero
reasoning tokens, and zero forced exits. The preregistered fresh-seed formal run is
therefore permitted.
