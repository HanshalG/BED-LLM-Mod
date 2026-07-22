# RockSample[15,15] Gemma 4 12B Seed-Robustness Result

**Status:** both preregistered fresh seeds independently passed every primary and
corroborating interval; the three-seed robustness claim passes.

The frozen non-thinking Gemma 4 12B K4/h2 protocol was repeated at seeds `24107`
and `24108` after the original seed `24106` result was known. Each fresh run used
30 paired trials, 15 rounds, exact 32,768-state Bayesian verification, and the
controls and endpoints frozen in
`ROCKSAMPLE_15_15_12B_VLLM_ROBUSTNESS_PREREGISTRATION.md`. No seed was replaced.

## Independent Decisions

Positive values favor StrategyEIG.

| Seed | Control | Entropy-AUC gain [95% CI] | Truth-log-AUC gain [95% CI] | W/T/L |
| ---: | --- | ---: | ---: | ---: |
| 24107 | Shared-roots d1 | +0.6584 [+0.6112, +0.7064] | +0.6940 [+0.6297, +0.7556] | 30/0/0 |
| 24107 | Exhaustive d1 width | +0.6330 [+0.5870, +0.6786] | +0.6862 [+0.5920, +0.7840] | 30/0/0 |
| 24107 | Random strategies | +0.6359 [+0.5791, +0.6911] | +0.5827 [+0.4820, +0.6779] | 30/0/0 |
| 24108 | Shared-roots d1 | +0.6482 [+0.6030, +0.6937] | +0.6113 [+0.5270, +0.6918] | 30/0/0 |
| 24108 | Exhaustive d1 width | +0.6102 [+0.5609, +0.6598] | +0.5378 [+0.4459, +0.6277] | 30/0/0 |
| 24108 | Random strategies | +0.6277 [+0.5832, +0.6741] | +0.6505 [+0.5604, +0.7404] | 30/0/0 |

Every lower bound is positive. StrategyEIG won all 180 fresh-seed paired
entropy-AUC comparisons, so the preregistered robustness gate passes without using
the pooled analysis to rescue either seed.

## Three-Seed Estimate

The registered equal-seed-weight stratified bootstrap is secondary.

| Control | Entropy-AUC gain [95% CI] | Truth-log-AUC gain [95% CI] | W/T/L |
| --- | ---: | ---: | ---: | ---: |
| Shared-roots d1 | +0.6417 [+0.6171, +0.6672] | +0.6360 [+0.5960, +0.6752] | 90/0/0 |
| Exhaustive d1 width | +0.6075 [+0.5813, +0.6340] | +0.5933 [+0.5422, +0.6446] | 90/0/0 |
| Random strategies | +0.6194 [+0.5930, +0.6451] | +0.6251 [+0.5718, +0.6773] | 90/0/0 |

Across seeds, the K4 proposal support captured 49.5%--53.3% of exhaustive d2
value, StrategyEIG moved on 551/1,350 decisions, and its remaining entropy-AUC gap
to exhaustive d2 was 0.4212--0.4666 nats. Thus 12B consistently supplies useful
non-myopic branches, though bounded K4 proposals do not recover the exact oracle.

## Execution Audit

- Jobs `106288` and `106289` completed on `oat14` in `msc`, with `oat12`
  excluded, from preregistration commit `b55c4e6`.
- Each fresh run made 1,290 physical requests and retained zero rejected responses.
  Including the original seed's reported crash-and-resume usage, the three audits
  account for 3,937 requests, 12,689,068 prompt tokens, and 960,235 completion
  tokens at zero API cost.
- All selected actions were legal; shared-root pairing, matched-width compute,
  random-K, and terminal checks passed. All three runs had zero reasoning tokens,
  forced exits, and rollout-scoring LLM calls.
- The independent auditor reconstructed all paired values and bootstrap intervals
  from stored traces. The pooled bootstrap used its frozen base seed `24109`.

The result upgrades the 12B finding from a successful transfer seed to a replicated
capacity result. Together with the failed E4B transfer, it brackets a practical
proposal-quality boundary for this interface; because model sizes were not paired
on identical hidden states, that cross-model boundary remains descriptive.
