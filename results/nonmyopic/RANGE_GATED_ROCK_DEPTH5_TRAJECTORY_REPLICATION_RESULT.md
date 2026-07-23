# Focused Range-Gated Rock H5 Trajectory Replication Result

Status: **all preregistered producer and independent-audit gates passed**.

## Primary Result

Three fresh 50-pair trajectory replications of the unchanged hierarchical h5
policy all passed separately. The preregistered fresh-only pool therefore contains
150 paired trajectories and excludes the original seed `24245`.

| Comparison | Pooled entropy-AUC gain | Independent 95% CI | W/T/L |
| --- | ---: | ---: | ---: |
| Same compiled plans scored at h4 | +.272658 | [.264946, .280007] | 150/0/0 |
| Matched-random h5 targets | +.211757 | [.201538, .221822] | 148/2/0 |
| Exhaustive receding-horizon d4 | +.284980 | [.279040, .290491] | 150/0/0 |

Truth-log-posterior AUC independently corroborates all three comparisons:

| Comparison | Pooled truth-log AUC gain | Independent 95% CI | W/T/L |
| --- | ---: | ---: | ---: |
| Same compiled plans scored at h4 | +.279295 | [.249504, .309840] | 144/0/6 |
| Matched-random h5 targets | +.212503 | [.193351, .232843] | 147/2/1 |
| Exhaustive receding-horizon d4 | +.297432 | [.265465, .329274] | 144/0/6 |

Pooled exact-d5 gap recovery, registered-route rate, and on-site-by-round-five
rate are all `1.0`.

## Per-Seed Replication

Every per-seed producer interval and every independently bootstrapped interval
has a positive lower bound.

| Seed | Entropy vs h4 | Entropy vs random h5 | Entropy vs exact d4 | Recovery | Physical/logical |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 24250 | +.278210 | +.193525 | +.278210 | 1.0 | 18/400 |
| 24252 | +.295769 | +.232841 | +.295769 | 1.0 | 18/400 |
| 24254 | +.243994 | +.208905 | +.280962 | 1.0 | 15/400 |

The h5 policy selected the registered route and reached an on-site rock-6 check
by round five on all 150 trials. The lower h4-control gain on seed `24254`
reflects later-history target variation, not a route or mechanics failure; that
seed still passes its independent h4 interval at `[.225730, .261188]`.

## Audit And Serving

- Per-seed independent audit seeds: `24251`, `24253`, `24255`.
- Pooled producer/audit bootstrap seeds: `24256`, `24257`.
- All source priors, truths, common observation uniforms, histories, compiled
  plans, exact values, controls, posteriors, paired values, aggregates, and
  fresh intervals replayed.
- All 1,200 logical decisions were accounted by 51 physical prompt identities.
- No LLM call occurred inside exact scoring or audit.
- All 51 reasoning passes reached the registered length exit and all 51 bounded
  finalizations succeeded.
- Zero target responses were invalid and no validation retry was needed.
- Four R1, two R2, and one R3 Wafer-style provider notices were normalized by
  the previously validated generic adapter repair.

OpenRouter cost was `$0.09689739`, moving project spend from
`$40.39856628` to `$40.49546367` of the `$110` ceiling.

## Interpretation

The positive non-myopic result is not carried by one favorable trajectory seed
or one provider realization. Across three fresh seeds, semantic target generation
plus deterministic routing and exact h5 verification consistently beats both
the identical-plan h4 scoring control and matched-random h5 target generation.
The result remains scoped to an engineered exact finite task and is LLM-Modulo
evidence, not unaided language-model planning.
