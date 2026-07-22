# RockSample[15,15] Gemma 4 12B Transfer Result

**Status:** preregistered primary and truth-log transfer gates passed.

The dense, non-thinking Gemma 4 12B proposer completed the frozen 15-rock K4/h2
protocol at fresh seed `24106`. The experiment used 30 paired trials, 15 rounds,
the exact 32,768-state verifier, and the controls and endpoints frozen in
`ROCKSAMPLE_15_15_12B_VLLM_PREREGISTRATION.md`. It was not pooled with 26B or E4B.

## Registered Decision

Positive values favor StrategyEIG.

| Control | Entropy-AUC gain [95% CI] | Truth-log-AUC gain [95% CI] | AUC W/T/L |
| --- | ---: | ---: | ---: |
| Shared-roots d1 | +0.6186 [+0.5843, +0.6565] | +0.6026 [+0.5393, +0.6652] | 30/0/0 |
| Exhaustive d1 width | +0.5792 [+0.5411, +0.6202] | +0.5560 [+0.4776, +0.6418] | 30/0/0 |
| Random strategies | +0.5946 [+0.5628, +0.6276] | +0.6422 [+0.5517, +0.7356] | 30/0/0 |

All three entropy-AUC and all three truth-log-AUC lower bounds are strictly
positive. StrategyEIG won every paired entropy-AUC comparison, so both registered
gates pass.

## Capacity Result

StrategyEIG captured 49.5% of exhaustive d2 value over nonterminal h2 decisions and
moved on 173/450 decisions. Shared d1 captured 6.5%, matched width 19.2%, and matched
random strategies 17.2%; both myopic controls made zero moves. Terminal-objective
exhaustive d2 remained better by 0.4666 entropy-AUC nats [0.4284, 0.5028].

This brackets the E4B failure rather than supporting a 26B-only effect. E4B captured
6.8% of exhaustive d2 value and failed the registered transfer gate, whereas 12B
captured 49.5% and passed all six comparisons. The three successful 26B seeds
captured 41.1%--47.6%. These model runs use different fresh seeds, so coverage and
effect-size differences are descriptive, not paired causal model-size estimates.
The defensible conclusion is that exact verification needs a sufficiently capable
proposal model, and non-thinking 12B was sufficient under this K4 interface.

## Execution Audit

- Initial job `106227` failed closed before an endpoint after a floating-point
  disagreement between complementary and joint branch probabilities. Commit
  `a24c51f` made predictive probability use the posterior's joint normalizer and
  added a regression test.
- Resume job `106265` ran on `oat14` in `msc`, with `oat12` excluded, from commit
  `c6133c7`. It revalidated and reused 372 accepted cells under the unchanged seed
  and protocol.
- Direct vLLM 0.23.0 served `google/gemma-4-12B-it` from the official CUDA 12.9
  container in bfloat16 at temperature zero, with no reasoning mode and a
  2,048-token completion cap.
- The completed artifact contains 1,356 replayable accepted cells, zero rejected
  responses, zero terminal failures, and zero terminal-followup normalizations.
- Cumulative serving usage includes the unlogged crashing response: 1,357 physical
  requests, 4,440,025 prompt tokens, and 335,732 completion tokens, with zero
  reasoning tokens, forced exits, rollout-scoring LLM calls, or API cost.
- The independent auditor reconstructed every paired value and all bootstrap
  intervals from the stored traces and confirmed both passed gates.

The 12B transfer strengthens the LLM-Modulo result: a middle-capacity non-thinking
policy proposer can supply valuable non-myopic branches at 32,768 latent states,
while the exact external verifier performs all rollout scoring and Bayesian updates.
