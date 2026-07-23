# UCI Thyroid Projected-Utility GPT-5.4 Mini Confirmation Result

The fresh-seed projected-utility confirmation passed every frozen scientific,
mechanical, contribution, and independent-audit gate.

## Result

Fifty held-out patient rows were followed for eight paired actions under
projected-utility GPT depth two, matched-random policies on identical roots, exact
depth one, and exhaustive depth two.

| Endpoint | Mean gain | Registered paired 95% CI | Independent 95% CI | W/T/L |
| --- | ---: | ---: | ---: | ---: |
| Entropy AUC vs exact d1 | +0.202618 | [+0.143734, +0.254275] | [+0.145560, +0.255193] | 41/0/9 |
| Truth-log AUC vs exact d1 | +0.172327 | [+0.052469, +0.319184] | [+0.053127, +0.319534] | 44/0/6 |
| Entropy AUC vs matched random | +0.118530 | [+0.081343, +0.157828] | [+0.080045, +0.158252] | 38/5/7 |
| Truth-log AUC vs matched random | +0.065388 | [+0.018099, +0.134641] | [+0.017781, +0.130241] | 39/5/6 |

Projected utility recovered 99.44% of exhaustive depth two's mean entropy-AUC gain
over exact depth one and selected blood collection first on 50/50 trajectories.
Mean entropy traces by round were:

| Arm | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Projected utility | .3103 | .0774 | .0739 | .0671 | .0654 | .0317 | .0333 | .0333 |
| Exhaustive d2 | .3103 | .0774 | .0683 | .0671 | .0618 | .0317 | .0333 | .0333 |
| Exact d1 | .3053 | .2895 | .2896 | .2890 | .2866 | .2824 | .2836 | .2870 |
| Matched random | .3078 | .2884 | .2386 | .2076 | .1625 | .1731 | .1496 | .1128 |

Final class accuracy was .98 for projected utility and exhaustive d2, .94 for exact
d1, and 1.00 for matched random. Accuracy is not the registered sequential endpoint;
the paired entropy/truth-log AUC intervals establish the information-gathering gain.

## LLM contribution and projection

The run completed all 350 logical GPT cells using 384 physical requests. Twenty-six
invalid first responses were corrected by the registered retry. Eight cells remained
invalid after both responses and invoked projection:

- projected cells: 8/350 = 2.29%, below the frozen 5% ceiling;
- projected branches: 17/4,338 = 0.392%, below the frozen 1% ceiling; and
- non-projected branches: 4,321/4,338 = 99.61%.

Projection retained every valid branch and replaced only invalid or missing branches
with a legal minimum-entropy continuation. The independent audit replayed all 1,600
arm decisions, 1,976 unique exact planning subtrees, every posterior and aggregate,
and every projection event without LLM calls.

The first raw-float audit ordered nine mathematically zero-entropy branches
differently because values ranged around floating-point zero. The corrected audit
applied the model's pre-existing `EPSILON` before the preregistered legal-order
tie-break. The largest replacement's entropy excess was `2.33e-16`; no policy,
trajectory, endpoint, bootstrap, projection, or scientific threshold changed.

## Interpretation

This is a positive LLM-Modulo policy result, not an unaided language-model planning
result. The empirical model supplies calibrated branch-local utility cards; GPT
assembles almost all branch policies; exact rollout scoring chooses the root; and a
small, measured legal projection closes the serving gap. Compared descriptively with
the names-only fresh-seed run, recovery rose from 30.8% to 99.4%, collection from
2/50 to 50/50, and entropy gain versus matched random from -0.0339 to +0.1185.

S0 plus S1 used 397 physical requests, 3,049,175 prompt tokens, 46,891 completion
tokens, zero reasoning tokens or forced exits, and `$0.97310355`. Project spend is
`$36.18786751 / $110`, leaving `$73.81213249`.
