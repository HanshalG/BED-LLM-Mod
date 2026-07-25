# Animals Support-Expansion Development Result

Date: 2026-07-25

## Verdict

The preregistered 20-state development gate failed. The 60-target holdout was
not accessed and remains sealed.

All 20 states and 60 candidate rows completed. Expected regenerated support
size was dynamic on every state and selected a different root from immediate
EIG on all 20. Support expansion achieved realized truth coverage `.60`
versus `.55` for EIG, a mean paired gain of `+.05` with `2/17/1`
wins/ties/losses. Its model-averaged expected coverage was `.452784` versus
`.339153`, a gain of `+.113631`, and it recovered three targets absent from
the current generated support.

This is not a positive result. The paired bootstrap interval for realized
coverage gain was `[-.10, .20]`, and the exact one-sided sign-test value was
`.50`. Expected cardinality also lowered mean uniform truth mass conditional
on the realized branch from `.011133` for EIG to `.009833`.

Two frozen gates failed:

1. The public artifact reported support-score pairwise accuracy `.4766` versus
   `.5429` for EIG.
2. Two non-thinking generation responses reached the 4,096-token output cap
   and ended with `finish_reason=length`, violating the zero-forced-exit gate.

The pairwise implementation pooled candidate pairs across different states,
which is not a meaningful ranking comparison. A disclosed post-hoc correction
computes only within-state pairs: support expansion scores `.50`, EIG `.333`,
but only six pairs in three states have distinct binary outcomes. This
correction does not rescue the run because serving integrity independently
failed and the primary paired interval crosses zero.

## Post-Hoc Selector Audit

The preregistered support-retention control was the strongest development
selector. It achieved realized coverage `.65`, expected coverage `.437179`,
and four recoveries, versus EIG `.55`, `.339153`, and two recoveries. The
realized comparison was `2/18/0`; within-state pairwise accuracy was `.9167`
on the same sparse six pairs.

This selected support retention only after the endpoint was opened. It is not
confirmatory evidence. Across cached artifacts, retention is strongest on the
two matching stratified-prior developments (`+.0643` and `+.0980`
model-averaged coverage versus EIG), while older mismatched protocols are
mixed (`-.0074` to `+.0098`). A fresh holdout test must therefore be
preregistered separately and must not relabel this failed support-expansion
development.

## Serving And Artifacts

- Model: `google/gemma-4-26b-a4b-it`, non-thinking.
- Requests: `9,948`; retries: `0`; reasoning tokens: `0`.
- Cost: `$0.27802635`.
- Development SHA-256:
  `c6797b987b69c1abcc819f6b935c32436919da1863fd98388e9efc6f9f80e091`.
- Private raw SHA-256:
  `9727e7481d489f78935d5cd43591decda5ff11ecb11d2de5ab4d2784e83f9038`.
- Post-hoc audit SHA-256:
  `e1d7a32d879e2810a6985bd73e8524a060040f4ff10c07ef6225c54a7b8c8bf4`.
- Frozen implementation commit: `1d5d39d`.

No OatML resources were used.
