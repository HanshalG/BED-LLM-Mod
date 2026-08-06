# Paper Number Game Claim-Evidence Audit

Date: 2026-08-06

## Purpose

Audit the paper's headline LLM-native claims against hash-bound public artifacts
before the unopened August 7 control and August 8--9 confirmation. This audit
uses no model calls, opens no prospective endpoint, and does not change any
source-study status.

## Claim Map

| Claim family | Authoritative evidence | Supported ceiling |
| --- | --- | --- |
| Powered two-query policy | `number_game_depth_two_powered` in `paper/claim_manifest.json` | Dynamic predictive-risk BED beats myopic, fixed-support d2, random, and PTS on the preregistered 32-tree study. |
| Cross-planner policy robustness | `number_game_crossplanner_canonical_pooled128` and `number_game_crossplanner_first_link_mechanism128` | Retrospective 128-tree synthesis supports d3 over myopic by 12.48% and positive simulated-to-realized first-link calibration. It does not relabel source transport nulls. |
| Additional planning depth | `number_game_depth_three_uncertainty_gate128` | D3 versus d2 is null. The paper must not claim monotonic benefit from planning horizon. |
| Dynamic versus fixed support | `number_game_pooled_dynamic_vs_fixed64`, `number_game_qwen_dynamic_vs_fixed_resilient96_v2`, and the fully fresh source artifact | Retrospective cross-planner evidence is positive. The powered 96-tree Brier primary is positive but mechanics-qualified. The fully fresh source preserves d3 over myopic while dynamic d3 versus fixed d3 is null. |
| Matched answer conditioning | `number_game_qwen_history_blind_first_link_confirmation32` | A disjoint preregistered control passes its four registered gates: second-stage MSE and coverage improve, at least 20 roots change, and root-specific conditioning benefit predicts realized advantage. The selected-root mean contrast remains null and was not a success gate. |
| Pooled matched controls | `number_game_history_conditioning_pooled64` | Retrospective synthesis preserves second-stage MSE, coverage, and selective-calibration gains. Its selected-root mean remains null and source statuses remain unchanged. |
| Fully fresh causal replication | Fully fresh source result and pre-outcome claim classifier | Not established. The source dynamic-versus-fixed endpoint family already failed, so even a positive history-blind control can support only separate fresh policy and matched-conditioning evidence. |

## Editorial Decision

The paper can make a strong LLM-native policy claim because the LLM generates
the open-ended executable support and counterfactual belief trees that the
planner scores. It must keep that claim separate from the narrower causal claim
that branch-conditioned regeneration itself improves selected policy value.

Accordingly:

- the contribution list now calls the headline a powered LLM-native policy
  result;
- the Number Game subsection heading now names policy selection rather than
  saying proposal dynamics generally "works";
- the method paragraph states that planning over generated belief dynamics does
  not by itself identify support regeneration as the cause;
- the matched-prompt result remains a four-gate confirmation of delayed support
  quality and selective first-link calibration, with its selected-root mean null
  stated explicitly;
- the full fresh causal policy-mechanism tier remains unresolved and cannot be
  inferred from the currently banked results.

## Verification

After the wording change, the claim-manifest validator passed all 37 hash-bound
claim bundles. The draft validator compiled a six-page PDF with all 14 required
limitation topics and both required figures, and the focused validator tests
passed `12/12`.
