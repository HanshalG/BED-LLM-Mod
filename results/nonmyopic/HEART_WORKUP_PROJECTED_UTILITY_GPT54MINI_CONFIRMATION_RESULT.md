# Cleveland Heart Projected-Utility GPT-5.4 Mini Confirmation Result

The fresh paired confirmation passed every producer-side scientific, mechanical, and
contribution gate, and its independent replay passed every mechanical check.
However, one frozen independent-bootstrap corroboration criterion narrowly crossed
zero. The overall preregistered confirmation is therefore **not a full pass**.

## Paired trajectory result

Fifty Cleveland rows were followed for eight paired actions under projected-utility
GPT depth two, matched-random continuations, exact depth one, and exhaustive exact
depth two.

| Endpoint | Mean gain | Registered paired 95% CI | Independent 95% CI | W/T/L |
| --- | ---: | ---: | ---: | ---: |
| Entropy AUC vs exact d1 | +0.063140 | [+0.034180, +0.092438] | [+0.034747, +0.092558] | 25/14/11 |
| Truth-log AUC vs exact d1 | +0.071701 | [+0.027721, +0.122076] | [+0.028517, +0.123133] | 24/14/12 |
| Entropy AUC vs matched random | +0.028906 | [+0.006715, +0.052096] | [+0.006438, +0.052326] | 20/18/12 |
| Truth-log AUC vs matched random | +0.029052 | [+0.000110, +0.062868] | **[-0.000682, +0.062253]** | 19/18/13 |
| Earlier workup rounds vs d1 | +2.140000 | [+1.660000, +2.620000] | [+1.660000, +2.620000] | 36/14/0 |

The primary registered endpoint, entropy AUC, is positive against both exact depth
one and matched random under both bootstrap seeds. Truth-log AUC is robustly positive
against depth one. Against matched random, its registered lower bound is
`+0.000110`, but the independent fresh-bootstrap lower bound is `-0.000682`; this
single frozen audit gate prevents an overall pass.

Projected utility recovered `100.89%` of exhaustive depth two's mean entropy-AUC
gain over depth one. Its mean workup round was `3.18`. Mean post-action entropy
traces were:

| Arm | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Projected utility | .5546 | .5106 | .4264 | .2544 | .1662 | .0408 | .0127 | .0000 |
| Exhaustive d2 | .5546 | .5106 | .4264 | .2452 | .1810 | .0397 | .0127 | .0000 |
| Exact d1 | .5546 | .5065 | .4309 | .3276 | .3134 | .2741 | .0362 | .0277 |
| Matched random | .5546 | .5196 | .4318 | .3085 | .2474 | .1213 | .0139 | .0000 |

Final class accuracy was 1.00 for projected utility, exhaustive depth two, and
matched random, and .98 for exact depth one. Accuracy is descriptive and saturates;
it is not the registered sequential endpoint.

## LLM contribution and audit

All 350 logical GPT cells were valid on the first response. Projection authored
`0/350` cells and `0/3,028` branches. The independent implementation replayed every
history, state, deterministic observation, posterior metric, GPT policy, random
policy, exact action, aggregate, and projection count without an LLM call. Every
mechanical audit check passed.

The run used 350 requests, 1,243,956 prompt tokens, 11,871 completion tokens, zero
reasoning tokens or forced exits, and `$0.46055610`. Including S0 and the proposal
gate, the Cleveland projected-utility line cost `$0.58556910`. Project spend is
`$37.22902951 / $110`, leaving `$72.77097049`.

## Interpretation

The clean entropy result, essentially complete exact-depth-two recovery, zero
projection, and earlier workup ordering show that the utility-grounded LLM-Modulo
architecture transfers beyond thyroid to a second native semantic acquisition task.
The result does not satisfy the stronger preregistered claim that both entropy and
truth-log improvements over random remain positive under an independent bootstrap.
It should be reported as strong primary-endpoint transfer with marginal
truth-calibration corroboration, not as an all-gates confirmation.
