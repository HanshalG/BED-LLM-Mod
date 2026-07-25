# BrowseComp-Plus Semantic Mechanics v2 Result

## Preregistered Decision

Version 2 fails closed after all 55 model calls because only 7/10 terminal
beliefs satisfy the strict positive-weight grammar. There is no repair,
renormalization, response reuse, or v3. Development and holdout remain sealed.

- Initial / refresh / future-scorer parses: `5/5`, `30/30`, `10/10`.
- Terminal parses: `7/10`.
- Two terminal responses assign zero mass to some hypotheses; one response's
  positive weights do not sum to 100.
- Physical requests / HTTP attempts: `55 / 55`.
- Retries / reasoning tokens / forced exits: `0 / 0 / 0`.
- Cost: `$0.287345`.
- Public failure SHA-256:
  `6272008d7da3730ea77dddcf76d3330268ee3a2d765ea86544346ae9492fb1d2`.
- Private raw SHA-256:
  `73f93e58e64706332c60477363aeed0b3134387008dd80026acdeab485230dea`.
- OatML use: none.
- Public posthoc analysis SHA-256:
  `8c84ffd5afd14576607fdbfbeb95489a58d7447f4a2729c654578a545e6c6a36`.

## Exploratory First-Link Read

Because every initial, refresh, and scorer response parsed before the terminal
failure, a zero-call posthoc reconstruction evaluates the already
preregistered evidence-ranking quantities. This is diagnostic only and cannot
turn the run into a pass.

- Root beliefs changed: `30/30`.
- Adaptive queries differ from roots: `30/30`.
- Branches adding evidence at step two: `20/30`.
- Direct score vs immediate evidence: `0.5610` over `41` pairs.
- Future score vs future evidence gain: `0.4211` over `38` pairs.
- Full score vs total evidence: `0.4762` over `42` pairs.
- Direct score vs total evidence: `0.5714` over `42` pairs.
- Shuffled full score vs total evidence: `0.5476` over `42` pairs.
- Full minus direct-on-total: `-0.0952`.
- Full minus shuffled: `-0.0714`.
- Strategy-d2 differs from myopic on `1/5` tasks, with `0` evidence wins and
  `1` loss.

The semantic transition works, but the LLM future-value link does not. Future
scores are high and weakly discriminative even when realized future gain is
zero; adding them damages the useful direct ranking. The deranged path-belief
control outperforming aligned beliefs is additional evidence that the scorer
is not using root-specific refreshed support reliably. More trials or deeper
planning are not authorized for this method.
