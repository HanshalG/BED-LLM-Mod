# RockSample[15,15] GPT-5.4 Mini Cross-Family Result

**Status:** all three preregistered primary and all three truth-log corroboration
gates passed.

The non-thinking GPT-5.4 Mini proposer completed the frozen 32,768-state K4/h2
protocol at fresh seed `24114`. The exact finite-state component generated every
observation, updated every posterior, and scored every rollout branch.

## Registered Decision

Positive values favor StrategyEIG.

| Control | Entropy-AUC gain [95% CI] | Truth-log-AUC gain [95% CI] | AUC W/T/L |
| --- | ---: | ---: | ---: |
| Shared-roots d1 | +0.4716 [+0.3762, +0.5601] | +0.5236 [+0.3965, +0.6481] | 28/0/2 |
| Exhaustive d1 width | +0.4562 [+0.3653, +0.5482] | +0.5553 [+0.4003, +0.7054] | 28/0/2 |
| Random strategies | +0.4410 [+0.3245, +0.5483] | +0.5276 [+0.3984, +0.6509] | 27/0/3 |

Every registered lower bound is strictly positive. The result therefore extends
the hardest-scale positive sign from Gemma to a second model family without
pooling models, seeds, or serving paths.

## Mechanism

- StrategyEIG moved on 233/450 decisions; both myopic controls made zero moves.
- K4 proposals captured 45.7% of exhaustive d2 value on nonterminal h2 rounds.
- StrategyEIG's entropy-AUC difference from terminal-objective exhaustive d2 was
  `-0.0714` `[-0.1623,+0.0208]`, statistically unresolved at this sample size.
- The result is not merely wider search: it beats the compute-matched exhaustive
  d1 width control and the matched random-strategy control.

The proposal fraction lies inside the 41.1%--47.6% range of the three successful
26B seeds and below the 49.5%--53.3% range of the three 12B seeds. Those contrasts
are descriptive because each model used different fresh seeds. The valid claim is
cross-family replication of the positive sign, not model ranking.

## Execution Audit

- Serving smoke: 10/10 accepted, zero rejects/reasoning/forced/terminal failures,
  `$0.03948225`.
- Formal: 1,312 accepted cells from 1,372 physical requests after 60 bounded
  rejected attempts; no cell exhausted its repair allowance.
- Usage: 3,898,276 prompt tokens, 252,827 completion tokens, zero reasoning tokens
  and forced exits; `$3.68507010`, below the frozen `$6.00` hard cap.
- All legality, pairing, shared-root, width-compute, random-K, and terminal
  mechanics passed. Exact rollout scoring made zero LLM calls.
- The independent auditor reconstructed every paired value and all bootstrap
  intervals from stored traces before this report was written.

Together with the 7-, 11-, and now 15-rock GPT replications, this shows that the
StrategyEIG gain is not tied to Gemma. Together with the three-seed 12B and 26B
results at 15 rocks, it also separates model-family robustness from the E4B
capacity failure.
