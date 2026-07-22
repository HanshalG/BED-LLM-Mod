# RockSample[15,15] Direct-vLLM Replication Result

**Status:** preregistered primary and corroboration gates passed.

The direct-vLLM Gemma 4 26B A4B replication extends the positive LLM-policy
StrategyEIG result to the frozen 15-rock diagnosis geometry and its exact 32,768-state
belief. The 30 paired trials used seed `24101`, 15 rounds, K4 branch policies, horizon
two, and the controls and endpoints frozen in
`ROCKSAMPLE_15_15_VLLM_REPLICATION_PREREGISTRATION.md`.

## Registered Comparisons

Positive values favor StrategyEIG. Every primary entropy-AUC interval and every
truth-log-posterior-AUC interval excludes zero.

| Control | Entropy-AUC gain [95% CI] | Truth-log-AUC gain [95% CI] | AUC W/T/L |
| --- | ---: | ---: | ---: |
| Shared-roots d1 | +0.6723 [+0.6328, +0.7124] | +0.6088 [+0.4578, +0.7488] | 30/0/0 |
| Exhaustive d1 width | +0.6700 [+0.6301, +0.7111] | +0.5837 [+0.4263, +0.7249] | 30/0/0 |
| Random strategies | +0.6345 [+0.5722, +0.6877] | +0.5448 [+0.4139, +0.6640] | 30/0/0 |

The primary gate required all three entropy-AUC lower bounds to be positive; it
passed. The separately required truth-log corroboration also passed. The result is
therefore not explained by reusing better LLM roots in the d1 control, adding more
one-step exact scoring width, or merely sampling non-myopic branch policies at random.

## Mechanism

StrategyEIG moved on 150/450 decisions. It moved in every trial during rounds 1--3
and 5, then used the more accurate checks those movements enabled. Shared-roots d1
and exhaustive d1 width moved on 0/450 decisions. The matched random-strategy arm
moved 144 times but did not choose useful movement/check continuations consistently;
StrategyEIG beat it by +0.6345 entropy-AUC nats.

The K4 candidate set captured 41.1% of exhaustive d2 value over nonterminal
horizon-two decisions. This is lower proposal coverage than on the smaller maps, but
it was sufficient for a large deployed gain over all registered controls. It also
leaves a clear proposal-quality target for future work.

StrategyEIG had a +0.0671 entropy-AUC advantage over terminal-objective exhaustive d2
[+0.0280, +0.1073], with 21/0/9 paired wins/ties/losses. This does not make the K4
search more exhaustive: exhaustive d2 achieved lower final entropy, 8.9069 versus
9.3326, a 0.4258-nat advantage. The difference is objective timing. Exhaustive d2
spent an additional all-trial movement at round 6 to maximize its rolling two-step
terminal gain, whereas StrategyEIG began informative checks sooner and therefore had
better entropy AUC. This is consistent with the paper's separate rolling-horizon
alignment diagnosis and is not reported as oracle domination.

## Execution Audit

- Cluster job `106113` ran on `oat14` in `msc`, with `oat12` excluded.
- Direct vLLM served `google/gemma-4-26B-A4B-it` in bfloat16 with temperature zero,
  no reasoning mode, and a 2,048-token completion cap.
- The run accepted all 1,320 logical cells. One malformed JSON response was repaired,
  for 1,321 physical requests total; there were no terminal cell failures or resumes.
- Usage was 4,263,484 prompt tokens and 305,264 completion tokens, with zero reasoning
  tokens, forced exits, rollout-scoring LLM calls, or monetary API cost.
- The independent auditor reconstructed every paired value and all 10,000-replicate
  bootstrap intervals from the stored traces. Its initial run exposed and then fixed
  an auditor-only seed-component mismatch; the corrected audit exactly matches the
  runner and is covered by a non-degenerate regression test.

This is a fresh serving-backend replication under seed `24101`. It does not replace
the separately preregistered OpenRouter seed-`24100` confirmation, which remains
without an endpoint because the provider account exhausted its credits.
