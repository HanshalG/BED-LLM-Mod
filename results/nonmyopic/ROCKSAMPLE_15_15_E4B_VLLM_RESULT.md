# RockSample[15,15] Gemma 4 E4B Transfer Result

**Status:** preregistered primary and truth-log transfer gates failed.

The smaller, non-thinking Gemma 4 E4B proposer completed the frozen 15-rock K4/h2
protocol cleanly, but it did not reproduce the 26B A4B result. The 30 paired trials
used fresh seed `24105`, 15 rounds, the unchanged exact 32,768-state verifier, and
the controls and endpoints frozen in
`ROCKSAMPLE_15_15_E4B_VLLM_PREREGISTRATION.md`. The seed was not replaced and is not
pooled with 26B.

## Registered Decision

Positive values favor StrategyEIG.

| Control | Entropy-AUC gain [95% CI] | Truth-log-AUC gain [95% CI] | AUC W/T/L |
| --- | ---: | ---: | ---: |
| Shared-roots d1 | +0.0687 [+0.0254, +0.1208] | +0.0753 [-0.0045, +0.1602] | 19/0/11 |
| Exhaustive d1 width | +0.0408 [-0.0035, +0.0884] | -0.0029 [-0.0871, +0.0814] | 17/0/13 |
| Random strategies | +0.0518 [+0.0010, +0.1098] | +0.1076 [+0.0285, +0.1939] | 18/0/12 |

The primary gate required every entropy-AUC lower bound to be positive; the matched
width interval crosses zero. The separate truth-log gate fails against both shared
d1 and width. The small positive entropy gains against shared d1 and random are real
within this seed, but they do not satisfy the registered transfer claim.

## Capability Boundary

The failure localizes to the proposed K4 support. E4B StrategyEIG captured only 6.8%
of exhaustive d2 value over nonterminal h2 decisions, essentially the same as its
shared-d1 roots (6.8%) and below matched width (19.2%) and matched random strategies
(17.7%). It moved on 121/450 decisions, so the failure is not merely refusal to use
the enabling action. Its entropy-AUC gap to exhaustive d2 was -0.5517
[-0.5884, -0.5112].

By contrast, the three preregistered 26B direct-vLLM seeds covered 41.1%, 41.2%, and
47.6% of exhaustive d2 value and passed all 18 entropy/truth comparisons. Those runs
use different fresh seeds, so this is a descriptive capacity contrast rather than a
paired causal model-size estimate. Together with the E4B actual-prompt smoke's 9.6%
coverage, it shows a consistent missing-proposal failure: exact verification can
select the best supplied branch but cannot recover valuable branches absent from K4.

## Execution Audit

- Cluster job `106129` ran for about 65 minutes on `oat16` in `msc`, with `oat12`
  excluded, from preregistration commit `d0b0645`.
- Direct vLLM served `google/gemma-4-E4B-it` in bfloat16 at temperature zero, with
  no reasoning mode and a 2,048-token completion cap.
- The run accepted 1,289 cells with five bounded rejected responses, 75 deterministic
  terminal-followup normalizations, zero terminal cell failures, and no resume.
- It used 4,104,501 prompt and 323,884 completion tokens over 1,294 physical requests,
  with zero reasoning tokens, forced exits, rollout-scoring LLM calls, or API cost.
- The independent auditor reconstructed every paired value and all bootstrap
  intervals from the stored traces and confirmed both failed gates.

This result narrows the positive claim: the LLM-Modulo architecture transfers across
seeds and serving paths with the 26B proposer, but the hardest map still requires a
proposal model capable of placing valuable contingent branches inside the bounded
candidate set.
