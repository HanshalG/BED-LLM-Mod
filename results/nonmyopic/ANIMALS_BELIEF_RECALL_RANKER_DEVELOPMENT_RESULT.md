# Animals Belief-Recall Ranker Development Result

Status: **promising development result; not confirmatory policy evidence**.

## Result

The frozen non-thinking Gemma 4 26B ranker scored the 20 previously observed
coverage-dynamics states without receiving a target or truth-coverage field.

| Selector | Mean selected expected truth coverage | Mean regret | Active-state regret |
| --- | ---: | ---: | ---: |
| Immediate EIG | .127050 | .129029 | .286731 |
| Belief-recall ranker | .202346 | .053733 | .119407 |
| Candidate oracle, measurement only | .256079 | 0 | 0 |

The ranker's paired gain over immediate EIG was `+.075296`, with `4/14/2`
wins/ties/losses. Its candidate-level Spearman association with hidden expected
truth coverage was `.1244`, compared with `.0032` for immediate EIG.

The direction held on both source traces:

| Source seed | Paired coverage gain | W/T/L | Active ranker / EIG regret |
| ---: | ---: | ---: | ---: |
| 1304 | +.105167 | 3/6/1 | .072500 / .282833 |
| 1305 | +.045425 | 1/8/1 | .178042 / .291604 |

Only 9/20 states had nonzero within-pool coverage spread, so ties dominate.
This is exactly why a larger fresh holdout is required.

## Controls And Interpretation

The improvement is not explained by retaining the largest support. Reanalysis
of the same frozen rows showed that expected regenerated support size selected
coverage `.144613`, only slightly above immediate EIG and well below the
ranker's `.202346`.

The ranker appears to combine branch size with semantic enumerability. For
example, it preferred `Is it larger than a dog?` over mammal/water questions
in the Tiger state and `Is it a carnivore?` in the held-out Wolverine trace,
matching the candidates that caused the production generator to recover the
missing targets. It still made two harmful choices, so this is a noisy model,
not an oracle.

Targets in these two source files were inspected before the scoring prompt was
frozen. The result can motivate and size a holdout but cannot support a paper
claim or policy deployment.

## Serving And Cost

The strict v1 parser failed closed after all 20 responses because Gemma wrapped
JSON in Markdown fences; no ranking endpoint was produced. The preregistered
format-only v2 parser accepted one standard fence and retained the unchanged
scores.

- v1: 20 requests, 9,938 prompt + 463 completion tokens, `$0.00112158`.
- v2: 20 requests, 9,936 prompt + 471 completion tokens, `$0.00107490`.
- Both: zero reasoning tokens, forced exits, or forced finals.
- Project spend after development: `$40.51745853` of `$110`.

Raw artifacts are under
`results/nonmyopic/animals_belief_recall_ranker/20260723_development*`.
