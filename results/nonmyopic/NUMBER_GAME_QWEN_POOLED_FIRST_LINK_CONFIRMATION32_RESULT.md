# Number Game Qwen Pooled First-Link Confirmation-32 Result

Run: `number-game-qwen-pooled-first-link-confirmation32-20260729T094455Z`

Status: **gated null with positive average policy efficacy and null first-link
rank calibration**.

## Registered Outcomes

Depth three changes the myopic root on 30/32 fresh trees. On those changed
roots:

- mean simulated Brier advantage: `0.0239237`;
- mean realized Brier advantage: `0.0117922`;
- tree-bootstrap 95% interval: `[0.0046866, 0.0189202]`;
- wins/ties/losses: `23/0/7`;
- simulated-to-realized Spearman: `-0.0367075`;
- Spearman interval: `[-0.390851, 0.307441]`.

The changed-root count, realized mean, realized interval, and win-loss gates
pass. Both preregistered Spearman gates fail. Pooling therefore preserves a
strong average non-myopic selection effect but does not calibrate the
cross-tree magnitude of simulated advantage.

## Policy Efficacy

All three preregistered depth-three-versus-myopic Brier gates pass:

- depth-three Brier: `0.104586`;
- myopic Brier: `0.115641`;
- relative reduction: `9.56%`;
- paired tree-bootstrap difference:
  `[-0.0179124, -0.0044109]`;
- tree wins: `23/32`.

The Hamming and truth-coverage diagnostics regress directionally, so the
positive claim is limited to the registered Brier endpoint.

Depth three improves Brier over cross-fitted depth two by `4.16%`, with
16 tree wins, but the interval is
`[-0.0092045, 0.0000075]`. The upper endpoint remains positive, so monotonic
planning depth is still a null. Fixed-support depth three is similarly
imprecise.

## Mechanics

The run completed all 32 fresh trees and exactly 3,424 accepted requests for
`$4.09288128`, with zero reasoning tokens, forced exits, or item-salvaged
draws. All 3,424 provider outputs were strict JSON.

The composite mechanics gate fails in two places:

- 50 transparent provider retries exceed the frozen cap of 24;
- generated-only branch minima fall below eight.

The latter does not indicate deployed posterior collapse. Pooled initial
supports have minimum 26, retained first supports minimum 12, and retained
second supports minimum 9; all deployed-support gates pass. Generated-only
second-step minima are below eight on 20/32 trees, showing that retained
rejuvenation remains essential even after pooling.

## Interpretation

Independent pooling solves the single-draw support-collapse problem at the
deployed-policy level and produces another fresh, statistically positive
depth-three-over-myopic Brier result. It does not solve the registered
first-link *ranking* problem: raw simulated margins are not comparable across
trees even though their selected roots are beneficial on average. The
within-tree root-ranking diagnostic remains positive
(`0.307` mean Spearman versus `0.073` for depth two), which is compatible with
a scale-calibration rather than root-selection failure.

The registered status remains `gated_null`; no gate is removed, rounded, or
reclassified.

Public `RESULT.json` SHA-256:
`cef6ded08050e6df07de62bbbf548635f229a8bf73fac149b4d0bd960f7b0253`.

Public `TREES.json` SHA-256:
`78bbc36bab604c78e31a68e50d31a1f9d92bea02bd295b1f983fe2f2a96a4d68`.

Public `TARGETS.json` SHA-256:
`ad7df72b9596f1a196717971a498e67cc90b5608457e81b4bb3af237ffc7bb7d`.

Private raw-response SHA-256:
`522ae568b7856f1fb4f8614bf69bef92357276954f8e9ebcbd3c7e71766ec349`.
