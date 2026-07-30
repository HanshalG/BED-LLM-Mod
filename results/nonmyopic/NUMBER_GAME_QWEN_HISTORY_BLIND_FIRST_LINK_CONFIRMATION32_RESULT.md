# Number Game Qwen History-Blind First-Link Confirmation-32 Result

Date completed: 2026-07-30

## Decision

**The disjoint confirmation passes every preregistered mechanics and
scientific gate.**

The registered first-link relationship replicates: across the 21 trees where
the stored dynamic and fixed-support planners choose different roots, the
root-specific benefit of answer-conditioned support predicts the realized
advantage of dynamic selection at Spearman `0.5299`, with 95% interval
`[0.0579, 0.8272]`.

This confirms selective calibration of the path-dependent belief mechanism.
It does not establish a positive uniform selected-root mean effect, rescue the
V3 mean-contrast null, or change the mechanics-qualified status of the
96-tree source policy study.

## Direct Belief-State Effect

The confirmation uses source trees `32..63`, disjoint from V3's trees
`0..31`. Every branch receives two fresh Qwen3.7 Plus nonreasoning generations
from the initial no-observation prompt. Conditional and history-blind supports
then use the same validity filtering, consistency filtering, and recursive
parent retention.

After the first answer:

- conditional predictive MSE: `0.0416808`;
- history-blind predictive MSE: `0.0214811`;
- conditional-minus-blind difference: `+0.0201996`;
- conditional truth coverage is `0.5446` versus `0.4298`, or `+11.48` points.

After the second answer:

- conditional predictive MSE: `0.0345492`;
- history-blind predictive MSE: `0.0381814`;
- difference: `-0.0036323`, with 95% interval
  `[-0.0057956, -0.0013772]`;
- this is a `9.51%` relative MSE reduction;
- conditional truth coverage is `0.6122` versus `0.4397`, a `+17.25` point
  difference with interval `[16.47, 18.03]` points.

The first-stage calibration-for-coverage tradeoff and second-stage joint MSE
and coverage improvement independently reproduce V3's qualitative pattern.

## First-Link Confirmation

Dynamic and fixed-support depth-three policies choose different roots on
`21/32` trees, passing the frozen minimum-20 gate.

- mean conditioning benefit at the dynamic root: `0.0080517`;
- mean benefit at the fixed root: `0.0055899`;
- dynamic-minus-fixed mean contrast: `+0.0024618`;
- mean-contrast interval: `[-0.0106419, +0.0147061]`;
- contrast-to-realized-advantage Spearman: `0.5299`;
- Spearman interval: `[0.0579, 0.8272]`.

The mean contrast remains descriptively null, as it was in V3. The
preregistered confirmation claim is instead that relative conditioning
quality is calibrated to where dynamic root selection realizes an advantage.
That gate passes on a disjoint cohort after being discovered in V3.

## Mechanics And Cost

- exactly `32` trees, `1,536` branch slots, and `3,072` accepted requests;
- exactly `3,072` HTTP attempts, with zero retries or provider retries;
- all draws strict JSON and at least `16` valid rules;
- every pooled support has at least `24` valid rules;
- zero reasoning tokens and zero forced exits;
- second-draw novelty range `2..20`, mean `7.7051`, descriptive only;
- cost `$3.26475648`, below the frozen `$4.25` cap.

No V1, V2, or V3 response was reused. An independent zero-call replay from
the saved controls exactly reproduced all 32 scored trees and the complete
20,000-sample bootstrap.

A launcher-only preflight at `20260730T044103Z` failed before adapter
construction because the sourced key was not exported to Python. It opened no
formal seed and incurred no cost. The corrected environment export produced
the single paid execution reported here.

## Artifacts

Run:

`results/nonmyopic/number_game_qwen_history_blind_first_link_confirmation32/number-game-qwen-history-blind-first-link-confirmation32-20260730T044135Z`

- `RESULT.json` SHA256:
  `c47a6aba6c1ac5d9028f234c4670d62418c4ce8d5e09162145c188e6a2d2b725`
- `CONTROLS.json` SHA256:
  `c598d34dc8cbffb3a9c3b8fa2c69f0d9c21efffe929c3cf53fb8e0a45ec3b9e7`
- private raw responses SHA256:
  `b8ac8f398875f49836f3749a637f955adb1014a65b8d0bc85cd0f2b3faf9972d`
- run log SHA256:
  `c1c3de184dbe3c90ea39cb0b007bbb4e060df9b30587841c713d62cd31663e4f`
- zero-call launcher failure SHA256:
  `f97d9d4e684215f9d85ad0c1c1d356e915aee49658d10973015f764386e5480b`

Authenticated balance after the run was `$2.469132069`.
