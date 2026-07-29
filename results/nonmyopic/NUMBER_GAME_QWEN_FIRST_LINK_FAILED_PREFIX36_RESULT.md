# Number Game Qwen First-Link Failed-Prefix-36 Result

Run: `number-game-qwen-first-link-failed-prefix36-20260729T092018Z`

Status: **strong positive underpowered diagnostic; the 64-tree confirmation
remains failed closed**.

## Frozen Analysis

The analysis was registered after the prospective run failed during tree 37
but before any canonical endpoint was reconstructed or scored. It strictly
replays the 36 complete saved trees, makes zero model calls, and evaluates the
unchanged exact 33-concept endpoint. It has no pass threshold and cannot
rescue, replace, or reclassify the failed confirmation.

## First-Link Result

Depth three changes the myopic root on 31/36 trees. Among those changed roots:

- mean simulated Brier advantage: `0.0203659`;
- mean realized Brier advantage: `0.0106546`;
- tree-bootstrap 95% interval: `[0.0040415, 0.0176253]`;
- simulated-to-realized Spearman: `0.616532`;
- tree-bootstrap 95% interval: `[0.367286, 0.772509]`;
- wins/ties/losses: `21/0/10`.

Thus, in this fresh incomplete cohort, the simulator's ranking of
depth-three-versus-myopic root changes is directionally and quantitatively
aligned with the exact downstream endpoint.

## Policy Diagnostics

Across all 36 trees and 33 concepts, cross-fitted depth three has mean Brier
`0.106606` versus myopic EIG's `0.115781`, a relative reduction of `7.92%`.
The paired tree-bootstrap difference is
`[-0.0154015, -0.0033864]`, with 21 tree wins. The corresponding Hamming
reduction is `5.79%`, but its interval crosses zero.

Comparisons with deeper controls remain imprecise:

- versus cross-fitted depth two: `3.99%` Brier reduction,
  interval `[-0.0115934, 0.0018127]`;
- versus fixed-support depth three: `2.49%` reduction,
  interval `[-0.0081196, 0.0025992]`.

The candidate also improves Brier relative to PTS and random roots, but those
diagnostics do not establish monotonic planning-depth benefit.

## Interpretation

This is fresh, endpoint-sealed evidence for the first link needed by the
method: simulated non-myopic root advantage predicts realized downstream
advantage. It is stronger mechanistic evidence than another retrospective
reanalysis of already scored endpoints. However, the sample size is 36 rather
than the registered 64, the run stopped for transport before endpoint
scoring, and none of the original confirmation gates is evaluated. The
confirmation status therefore remains `failed_closed`.

Public `RESULT.json` SHA-256:
`9a483237336a5d80ed89a479e254ee80eef85ba96e17ebbf7f634bf227cca9de`.

Public `TREES.json` SHA-256:
`297ec94d28c2f6203d26427025bdd2032788722c815a8047b8b767f423c558ff`.
