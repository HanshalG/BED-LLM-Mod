# Number Game GLM Planner Depth-Three Preregistration

Date frozen: 2026-07-29, before the exact-10 GLM serving screen and before any
formal GLM response.

## Claim

Test the existing cross-fitted, twice-regenerated Number Game depth-three
policy with GLM 5.1 as a new planning generator and GPT-5.4 Mini as the
independent target, validation, and endpoint generator. Both are
nonreasoning. This run is authorized only by a complete pass of the separately
frozen exact-10 GLM serving screen.

## Frozen Design

- 32 trees, planning seeds `47000..47031`;
- target seeds `47100..47131`;
- eight validation draws per tree beginning at `47200`;
- 15 extra endpoint draws per tree beginning at `47500`, plus the target
  draw, for 16 endpoint draws per tree;
- 49 GLM planning and 24 GPT-5.4 Mini endpoint-family calls per tree;
- exactly 2,336 accepted responses;
- temperature `0.7`, explicit nonreasoning;
- retained-parent rejuvenation at both refreshes;
- independent cross-fitted root selection and shared endpoint evaluation;
- maximum reported cost and minimum starting balance `$7.50`.

All mechanics and confirmatory gates are byte-for-byte the Grok-family frozen
gates: support minima and novelty; at least 12 depth root changes; at least 1%
depth-three Brier gain with whole-tree CI below zero and at least 12 wins;
novel-target Brier nonregression; Brier intervals below zero against myopic,
fixed-support depth three, and PTS; and depth-three rank Spearman at least
`.7` and at least `.15` above depth two.

Brier is confirmatory because it is the optimized proper score. Hamming and
coverage remain fully reported secondary diagnostics. Any missing response,
parse failure, response replacement, selective removal, or post-response
change fails closed.
