# Number Game Planner Model Frontier Smoke

Date: 2026-07-29

## Question

Which currently available model is the best intelligence-cost choice for the
non-reasoning, executable-hypothesis-generation role in the Number Game
planner?

This is a serving and support-quality screen, not an efficacy comparison.
It uses the linked exact-10 retained-rejuvenation gate previously passed by
Qwen3.7 Plus. No target concepts or scientific endpoint are used.

## External Screen

Current OpenRouter prices and Artificial Analysis (AA) results identify three
plausible challengers:

| Model | OpenRouter input/output per 1M | Relevant AA result |
|---|---:|---|
| Qwen3.7 Flash | $0.03 / $0.13 | no independent result yet |
| DeepSeek V4 Flash | $0.14 / $0.28 | 29 non-reasoning; 37 high reasoning at $53.94 total evaluation cost |
| MiniMax M3 | $0.30 / $1.20 | 44 reasoning at $203.86 total evaluation cost |
| Qwen3.7 Plus | $0.32 / $1.28 | 39 reasoning at $311.21 total evaluation cost |
| GPT-5.4 Mini | $0.75 / $4.50 | 17 non-reasoning; 40 xhigh reasoning at $1,095.43 total evaluation cost |

Reasoning-mode AA scores are only a screening signal because the deployed
planner runs with reasoning disabled. For this role, strict schema adherence,
history consistency, extension diversity, and conditional novelty are more
directly relevant.

## Exact-10 Results

| Model | Status | Initial valid mean | Conditioned valid mean | Conditioned minimum | Cost | Total valid rules per dollar |
|---|---|---:|---:|---:|---:|---:|
| Qwen3.7 Plus | passed | 23.0 | 17.625 | 8 | $0.0103648 | 18,042 |
| DeepSeek V4 Flash | passed | 24.0 | 13.25 | 7 | $0.001933696 | 79,640 |
| MiniMax M3 | passed | 22.0 | 11.125 | 4 | $0.0088494 | 15,029 |
| Qwen3.7 Flash | failed closed | n/a | n/a | n/a | $0.000537 | 0 operationally valid |

DeepSeek and MiniMax each completed exactly ten accepted and ten HTTP
requests with zero retries, provider retries, reasoning tokens, or forced
exits. Both passed every linked retained-support threshold.

Qwen3.7 Flash completed all ten transport calls but ignored the required
schema and returned a top-level `concepts` list. The parser failed closed.
No model-specific parser or salvage path was added.

## Decision

The current task-specific Pareto set has two unresolved points:

- Qwen3.7 Plus is the proven efficacy and conditional-diversity point.
- DeepSeek V4 Flash is the cost point: 5.36 times cheaper per exact-10 gate
  and 4.41 times more operationally valid rules per dollar.

MiniMax M3 is dominated in this direct-generation role so far: its gate cost
is close to Qwen, while its conditional valid-rule yield is lower. Qwen3.7
Flash is not operationally usable under the frozen interface. GPT-5.4 Mini
is not the default scale model because existing Number Game cohorts do not
show an efficacy advantage commensurate with its roughly 2.5 times higher
cohort cost.

Do not switch the paper-critical planner to DeepSeek on this smoke alone.
The next authorized model-choice experiment is a paired downstream-risk gate
on a frozen canonical target and validation bank. DeepSeek must preserve the
existing depth-three-over-myopic effect and satisfy a predeclared quality
tolerance before it can replace Qwen for scaled runs.

## Artifacts

- Qwen reference RESULT SHA256:
  `89d4917fa7cd861a5bf264758d721365a17498cfb1a7bd75087287ec758ab997`
- DeepSeek RESULT SHA256:
  `5342c72b3eb02e9038f7447dded89e9d726785169e9834335f09da91a03b6b71`
- MiniMax RESULT SHA256:
  `b0489c2e32fc6591f01af9a9e697f7af69919927130f4b20e863ea95d12d55ad`
- Qwen Flash FAILURE SHA256:
  `b78096464f088482f0045cf67db80204ac44e83e67a325de28947ee3960c152f`
- Qwen Flash private raw SHA256:
  `5e54c8998b458caf8cec9b704885c7decdeda887b58b8e2eff5669af79547bdc`
