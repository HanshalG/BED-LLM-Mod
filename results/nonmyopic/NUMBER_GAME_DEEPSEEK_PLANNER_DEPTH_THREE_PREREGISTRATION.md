# Number Game DeepSeek Planner Depth-Three Preregistration

Date frozen: 2026-07-29, before the exact-10 serving smoke and before any
formal response under this protocol.

## Claim

Test whether the cross-fitted, twice-regenerated Number Game depth-three
policy improves the proper score it optimizes when the planning generator is
a third model family.

DeepSeek V4 Pro generates the initial and branch-conditioned planning
supports. GPT-5.4 Mini independently generates targets, eight validation
supports, and 16 endpoint supports per tree. Both are nonreasoning. Targets
and endpoint draws are hidden from planning.

This is not a repair or continuation of the failed Qwen formal run. It uses
a different planner, a different endpoint provider, fresh seeds, a separately
frozen transport gate, and a new run directory.

## Authorization

The run is authorized only if the exact-10 result from
`NUMBER_GAME_DEEPSEEK_PLANNER_SERVING_SMOKE_PREREGISTRATION.md` passes every
gate. Smoke efficacy is neither computed nor inspected.

## Frozen Design

- 32 trees, planning seeds `43000..43031`;
- target seeds `43100..43131`;
- eight validation draws per tree beginning at `43200`;
- 15 extra endpoint draws per tree beginning at `43500`, plus the target
  draw, for 16 endpoint draws per tree;
- 49 DeepSeek planning calls and 24 GPT-5.4 Mini target/validation/endpoint
  calls per tree;
- exactly 2,336 accepted responses;
- temperature `0.7`, explicit nonreasoning mode;
- retained-parent rejuvenation at both refreshes;
- eight cross-fitted validation supports select depth-three and depth-two
  roots;
- the same 16 independent endpoint supports evaluate every policy;
- maximum reported run cost `$5.00`;
- minimum starting OpenRouter balance `$5.00`.

The policies, parsers, support filters, candidate roots, PTS control,
cross-fitted scoring, endpoint aggregation, and whole-tree bootstrap are
unchanged from the frozen Qwen-family protocol.

## Confirmatory Gates

Every mechanical and scientific gate must pass:

1. Exact request accounting, at most eight retries, zero reasoning tokens,
   zero forced exits, and cost within budget.
2. Every initial support has at least 16 valid rules; every validation and
   endpoint support has at least 16; every retained first branch has at least
   eight; every retained second branch has at least four.
3. Every tree has at least 128 endpoint hypotheses novel to its initial
   support.
4. Cross-fitted depth three and depth two choose different roots on at least
   12 trees.
5. Depth three improves endpoint posterior-predictive Brier over equally
   cross-fitted depth two by at least 1%, its whole-tree bootstrap difference
   interval lies below zero, and it wins at least 12 trees.
6. Novel-target endpoint Brier does not regress.
7. Depth three has a negative Brier difference with an interval below zero
   against myopic EIG, fixed-support depth three, and PTS.
8. Mean depth-three source-to-endpoint Brier Spearman is at least `0.7` and
   exceeds depth two by at least `0.15`.

Posterior-predictive Brier is the confirmatory loss because it is the
explicit policy objective. Best-rule Hamming and exact-extension coverage
are reported as secondary diagnostics, including their paired intervals and
novel-target values, but are not silently promoted into a conjunction over
different losses.

Any parse failure, missing tree, post-response threshold change, selective
tree removal, or response replacement fails the run closed.
