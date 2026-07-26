# tau-Knowledge Future-First Robustness Result

The frozen zero-call audit is null or adverse. Directly optimizing the
future-uplift difference does not stabilize the V3.1 policy.

Public artifact:

`results/nonmyopic/tau_knowledge_future_first_robustness/ANALYSIS.json`

SHA-256:

`137f85d2aa468a991faf2b23deb0ea78bcfd2df84435925521c5f6c5976dad6d`

## Frozen Measurements

| Metric | Result |
|---|---:|
| Scorer-retest block future-gain accuracy | `.5867` |
| Rank-ensemble block future-gain accuracy | `.5901` |
| Pooled future-gain accuracy | `.5884` over 588 pairs |
| Future-first root agreement across six runs | `.7300` |
| Mean future-first endpoint | `22.33` documents |
| Mean myopic endpoint | `26.83` documents |
| Mean raw-full endpoint | `30.67` documents |
| Mean frozen-random endpoint | `22.00` documents |
| Future-first minus myopic | `-4.50` documents |
| Future-first minus raw full | `-8.33` documents |

Future-first was below both myopic and raw full in every one of the six
replicate totals. Averaging within task across scorer runs gave:

- versus myopic: 4 wins, 7 ties, 9 losses, mean `-.225`,
  one-sided exact sign-flip `p=.8765`;
- versus raw full: 1 win, 8 ties, 11 losses, mean `-.4167`,
  `p=.99976`.

All hash, shape, task-order, exact pair-value, and frozen-random reproduction
checks passed. The analysis made zero model calls and cost `$0`.

## Interpretation

The subtraction is not meaningless: its root selections are reproducible and
its future-only ranking remains above chance in both independent three-run
blocks. The failure is objective composition. A future-first policy sacrifices
too much immediate required-document value, while the raw full score preserves
both terms and is substantially better.

Therefore `full - myopic` remains a useful diagnostic for whether semantic
lookahead contains future signal. It must not be promoted to a standalone BED
acquisition objective. This same-task stabilization rule is closed and neither
strong nor directional criteria pass.
