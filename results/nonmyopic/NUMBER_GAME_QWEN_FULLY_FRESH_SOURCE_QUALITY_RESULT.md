# Number Game Fully Fresh Source-Quality Diagnostic Result

Date: 2026-08-06

Source: `number-game-qwen-fully-fresh-daily-stages-20260806T000200Z`

Audit: `number-game-qwen-fully-fresh-source-quality-20260806T004500Z`

## Status

**Retrospective partial mechanism positive with a realized-calibration null.**
The newly generated support is a significantly better approximation to the
exact canonical posterior after two answers, and the dynamic-selected roots
have more refresh-quality gain than the fixed-selected roots. However, that
tree-level quality contrast does not predict realized dynamic-versus-fixed
advantage.

This zero-call diagnostic does not rescue or relabel the source `gated_null`,
and it does not alter the mandatory later-day history-blind control.

## Canonical Posterior Approximation

| Stage | Fixed MSE | Dynamic MSE | Dynamic minus fixed 95% CI | Fixed coverage | Dynamic coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| After one answer | `.022385` | `.040649` | `[+.015732,+.021100]` | `.4176` | `.5540` |
| After two answers | `.036382` | `.033123` | `[-.006432,-.000141]` | `.4176` | `.6230` |

The first refresh broadens support from `18.16` to `36.97` unique extensions
and raises truth coverage by 13.64 points, but predictive MSE worsens. After
two answers, dynamic support has `39.05` extensions versus `9.61` fixed,
raises truth coverage by 20.54 points, and reduces canonical posterior MSE by
8.96%. Thus the fresh generator produces a better broad second-stage belief
state on average, not merely more hypotheses.

## Selected-Root Quality

Across the `29/32` changed-root trees:

- mean fixed-minus-dynamic quality gain at the dynamic-selected root:
  `.010245`;
- mean gain at the fixed-selected root: `.003633`;
- paired contrast: `.006613`, 95% interval
  `[.001195,.011495]`;
- Spearman between that contrast and realized fixed-minus-dynamic root Brier:
  `-.162`, 95% interval `[-.521,+.229]`.

The planner therefore tends to select roots where regeneration improves the
global canonical belief approximation more than at the fixed root. What fails
is the next link: variation in that improvement does not identify where the
selected dynamic root will beat the fixed root on independent target Brier.
This explains how the source can improve support quality yet achieve only an
imprecise 2.85% dynamic-versus-fixed policy gain.

## Cross-Cohort Reading

The earlier 96-tree diagnostic showed the complementary pattern:

| Cohort | Globally better second-stage dynamic MSE | Selected-root quality contrast CI positive | Quality contrast predicts realized advantage |
| --- | --- | --- | --- |
| Prior 96 trees | no | no | yes, `rho=.472 [.262,.645]` |
| Fully fresh 32 trees | yes | yes | no, `rho=-.162 [-.521,.229]` |

No cohort establishes all links jointly. Generated beliefs can improve on
average, and their selective quality can sometimes track endpoint value, but
the coupling is not generation-robust. This is consistent with the broader
project diagnosis that more lookahead cannot repair a biased or unstable
semantic transition model.

## Stored-Pool Control And Provenance

The stored all-branch pool again matches routed dynamic support exactly after
consistency filtering. This is a structural identity, not evidence about
history conditioning. Only the already-authorized fresh no-observation
generation control can identify that causal effect.

- model calls / cost: `0 / $0`;
- OatML use: `0`;
- bootstrap seed / samples: `100900 / 20000`;
- result SHA-256:
  `eac41373b90afd6a521da54b9ed57a68941efefe719a6828fb9f5fc228449bc4`;
- independent deterministic replay: exact SHA-256 match;
- source status and all source/control authorizations: unchanged.
