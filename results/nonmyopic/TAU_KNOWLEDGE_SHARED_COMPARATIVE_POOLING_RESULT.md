# tau-Knowledge Shared Comparative Pooling Result

## Decision

The serving stage passed cleanly. The 20-task development run then passed all
transport, endpoint, stability, and directional gates except the preregistered
global pairwise-accuracy gain. The overall gate therefore **fails**, and no
fresh tau task was opened.

The result is nevertheless a useful directional first-link signal:
non-myopic pooled ranking selected roots with 37 exact oracle-tail documents
versus 31 for the compute-matched pooled myopic ranker and 25 for random.
That `+6` gain came from 5 wins, 12 ties, and 3 losses.

## Frozen Metrics

| Metric | Myopic | Non-myopic | Gate |
| --- | ---: | ---: | ---: |
| Global root pairwise accuracy | `.6673` | `.6453` | non-myopic `>=.62` and gain `>=.05` |
| Pairwise-accuracy gain |  | `-.0220` | **failed** |
| Selected-root oracle-tail total | `31` | `37` | gain `>=4`, passed |
| Mean rank-replicate agreement | `.7933` | `.6919` | non-myopic `>=.55`, passed |
| Oracle-tail total versus random |  | `+12` | `>=+5`, passed |
| Root decisions changed |  | `13/20` | `>=4`, passed |
| Wins / ties / losses |  | `5 / 12 / 3` | wins-losses `>=2`, passed |

All 499 endpoint-distinct root pairs were included in the frozen global
pairwise metric. The non-myopic ranking cleared its absolute `.62` gate but
did not improve over the unusually strong pooled myopic ranking.

## Top-Focused Post-Hoc Audit

The global metric weights every lower-ranked root pair even though only the
top root is executed. A zero-call descriptive audit therefore measured each
selected root against all endpoint-distinct alternatives:

| Metric | Myopic | Non-myopic |
| --- | ---: | ---: |
| Selected-root-versus-all accuracy | `.6214` | `.7453` |
| Oracle-optimal selected roots | `8/20` | `10/20` |
| Total selected-root regret | `20` | `14` |
| Mean selected-root regret | `1.00` | `.70` |

This supports a top-versus-tail failure diagnosis, but it does not repair the
frozen gate. The `+6` endpoint gain has exact one-sided task sign-flip
`p=.17969`; a 100,000-draw task bootstrap gives a total-gain interval
`[-4, 16]`. Individual non-myopic ranking replicates selected totals
`33 / 38 / 30`, while the frozen Borda policy selected `37`.

## Integrity And Cost

Serving:

- exact logical requests / HTTP attempts / retries: `12 / 12 / 0`
- prompt / completion / reasoning tokens: `106,095 / 420 / 0`
- forced exits: `0`
- cost: `$0.2715375`
- private raw SHA-256:
  `bc0e8e8bd34d7418fd1cea1a05c671ba0ea085f95bdb4c1e5c667d6dc640f3f1`

Development:

- exact logical requests / HTTP attempts / retries: `120 / 120 / 0`
- prompt / completion / reasoning tokens: `944,490 / 4,200 / 0`
- forced exits: `0`
- cost: `$2.227233`
- private raw SHA-256:
  `e50d2895949ccec75ec6e202e9ea79cd885e666dd0b04d5a50960d2e2bc0d21b`

Combined spend: `$2.4987705`. The provider credit endpoint lagged the completed
development charge; the conservative post-run balance is
`$29.997867594`. There is no fixed reserve.

## Next Method

Close the exact complete-ranking policy. A scientifically distinct successor
may use its frozen myopic and non-myopic winners only as a two-root shortlist,
then compare those complete future trees directly in both A/B orders:

- choose a challenger only when both swapped comparisons agree;
- fall back to the myopic winner on disagreement; and
- evaluate the same oracle-continuation first-link endpoint.

This targets the demonstrated top-root decision while prospectively controlling
position bias and avoiding irrelevant lower-rank comparisons. It remains
development on open tasks until it passes a separately frozen gate.
