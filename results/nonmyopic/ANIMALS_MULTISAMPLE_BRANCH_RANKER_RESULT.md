# Animals Multi-Sample Branch Ranker Development Result

Status: **development gate passed; fresh holdout authorized**.

The unchanged target-blind branch-content ranker was applied to the broader
four-list supports on the 20 inspected seed-24279 states.

| Selector | Selected expected truth coverage |
| --- | ---: |
| Immediate EIG | .095833 |
| Multi-sample branch ranker | .189246 |
| Candidate oracle, measurement only | .255704 |

The paired gain was `+.093413` with `3/17/0` wins/ties/losses. Candidate-level
Spearman association was `+.160950` for the ranker versus `-.088740` for
immediate EIG. Active-state regret was `.166146` versus `.399677`.

All 20 responses parsed, all 100 explicit payload checks passed, and no model
payload contained target fields, truth-derived fields, or immediate EIG.
Serving used 20 non-thinking requests for `$0.00214169`.

Every frozen development condition passed. Because targets and endpoints were
already inspected, this is not confirmatory evidence; it authorizes one fresh
sealed holdout of the exact multi-sample generator and branch-content scorer.

Artifacts are in
`results/nonmyopic/animals_multisample_generator_recall/gemma26b_seed24279/`.
