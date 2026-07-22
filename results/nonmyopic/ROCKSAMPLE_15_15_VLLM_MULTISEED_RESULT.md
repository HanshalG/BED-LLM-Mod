# RockSample[15,15] Direct-vLLM Multi-Seed Result

**Status:** both preregistered fresh seeds passed every primary and corroboration
gate. All three direct-vLLM seeds pass.

Seeds `24102` and `24103` repeated the seed-`24101` 32,768-state experiment with
the identical Gemma 4 26B A4B checkpoint, K4 branch-policy interface, horizon,
controls, exact simulator, and 30-pair design. No seed was replaced.

## Per-Seed Gates

Positive values favor StrategyEIG. The two fresh seeds independently pass all six
required entropy-AUC and truth-log-posterior-AUC gates.

| Seed | Control | Entropy-AUC gain [95% CI] | Truth-log-AUC gain [95% CI] | W/T/L |
| ---: | --- | ---: | ---: | ---: |
| 24102 | Shared-roots d1 | +0.6619 [+0.6302, +0.6937] | +0.6947 [+0.5499, +0.8346] | 30/0/0 |
| 24102 | Exhaustive d1 width | +0.6590 [+0.6256, +0.6917] | +0.6932 [+0.5508, +0.8339] | 30/0/0 |
| 24102 | Random strategies | +0.6857 [+0.6454, +0.7255] | +0.6684 [+0.5462, +0.7798] | 30/0/0 |
| 24103 | Shared-roots d1 | +0.7176 [+0.6816, +0.7520] | +0.6750 [+0.4647, +0.8496] | 30/0/0 |
| 24103 | Exhaustive d1 width | +0.7157 [+0.6790, +0.7512] | +0.6672 [+0.4468, +0.8404] | 30/0/0 |
| 24103 | Random strategies | +0.7076 [+0.6714, +0.7431] | +0.6705 [+0.4649, +0.8357] | 30/0/0 |

## Secondary Three-Seed Pool

The preregistration allowed a descriptive equal-seed-weight stratified paired
bootstrap only after both fresh decisions were fixed. Across seeds `24101`--`24103`:

| Control | Entropy-AUC gain [95% CI] | Truth-log-AUC gain [95% CI] | W/T/L |
| --- | ---: | ---: | ---: |
| Shared-roots d1 | +0.6839 [+0.6626, +0.7051] | +0.6595 [+0.5607, +0.7486] | 90/0/0 |
| Exhaustive d1 width | +0.6815 [+0.6603, +0.7036] | +0.6480 [+0.5479, +0.7369] | 90/0/0 |
| Random strategies | +0.6760 [+0.6493, +0.7017] | +0.6279 [+0.5373, +0.7082] | 90/0/0 |

Pooling does not rescue a seed: both fresh runs had already passed independently.

## Mechanism And Execution

- StrategyEIG moved on 150, 174, and 162 of 450 decisions under seeds
  `24101`, `24102`, and `24103`; both myopic controls moved zero times in every run.
- K4 captured 41.1%, 41.2%, and 47.6% of exhaustive d2 value over nonterminal
  horizon-two decisions. The gain persists despite substantial proposal headroom.
- StrategyEIG's entropy-AUC advantage over terminal-objective exhaustive d2 was
  +0.0671, +0.0686, and +0.0989 nats, while exhaustive d2 retained lower final
  entropy. The repeated timing pattern supports the rolling-objective diagnosis.
- The three runs accepted 3,960 logical cells in 3,964 physical requests, with four
  bounded malformed-response repairs, zero terminal failures, reasoning tokens,
  forced exits, resumes, API cost, or rollout-scoring LLM calls.
- Independent auditors reconstructed all raw paired values and all 18 per-seed
  entropy/truth bootstrap intervals before the secondary pool was computed.

This establishes fresh-seed robustness for one checkpoint and serving path. It does
not add a second model family at 15-rock scale.
