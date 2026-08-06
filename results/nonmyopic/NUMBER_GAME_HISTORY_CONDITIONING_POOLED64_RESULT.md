# Number Game History-Conditioning Pooled-64 Result

Date completed: 2026-08-06

## Decision

**Two disjoint matched controls establish a replicated delayed belief-state
effect and selective first-link calibration, but not uniform selected-root
uplift.**

This is a zero-call, retrospective synthesis. It preserves the development
cohort's gated-null status, the disjoint confirmation's passed status, and the
mechanics-qualified status of the source policy study.

## Delayed Belief-State Effect

The two source controls compare answer-conditioned Qwen generation with fresh
history-blind generation under matched calls, filtering, and recursive parent
retention. The synthesis pools 64 disjoint trees using cohort-stratified tree
bootstrapping.

| Stage | Conditioned MSE | History-blind MSE | MSE difference | Coverage difference |
| --- | ---: | ---: | ---: | ---: |
| After one answer | `.039851` | `.021127` | `+.018724` | `+10.97` points |
| After two answers | `.033532` | `.038106` | `-.004574` | `+16.92` points |

After two answers, conditioning reduces canonical posterior-predictive MSE by
12.0% relative to history-blind generation. The paired difference interval is
`[-.006153,-.002925]`, and the coverage-difference interval is
`[+.162879,+.175426]`. The repeated first-stage cost followed by second-stage
joint calibration and coverage gains is direct evidence that the LLM belief
transition depends usefully on the observed path; it is not an extra-call or
filtering effect.

## First-Link Boundary

Dynamic and fixed-support depth-three policies choose different roots on
`48/64` trees (`27` development, `21` confirmation). Across those trees:

- root-specific conditioning benefit predicts realized dynamic-selection
  advantage at Spearman `.6082`, with stratified-bootstrap interval
  `[.3537,.7913]`;
- mean conditioning benefit is `.009006` at the dynamic-selected root and
  `.007201` at the fixed-selected root;
- their mean contrast is only `+.001805`, with interval
  `[-.005594,+.008900]`.

The evidence therefore supports calibrated heterogeneity: dynamic planning
helps where answer conditioning is comparatively useful. It does not support
the stronger statement that the selected dynamic root receives a uniformly
larger conditioning benefit on average.

## Provenance

- source cohorts: trees `0..31` and `32..63` from the fresh powered Qwen study;
- bootstrap: cohort-stratified tree resampling, seed `10700000`, 20,000 draws;
- model calls / cost: `0 / $0`;
- source result hashes:
  `29bb76074dfb53a6efef352fde88fbca6c3a7dc591d0936d132c82050f1dc71d`
  and
  `c47a6aba6c1ac5d9028f234c4670d62418c4ce8d5e09162145c188e6a2d2b725`;
- pooled `RESULT.json` SHA-256:
  `186f4a6b6152a4a2435e5eda60d45fd1ab696767361d4bfae8ae95fa9aaa6db4`.
