# Number Game Grok Planner Depth-Three Preregistration

Date frozen: 2026-07-29, before the exact-10 serving smoke and before any
formal Grok response under this protocol.

## Claim And Authorization

Test whether cross-fitted, twice-regenerated Number Game depth three improves
the proper score it optimizes with Grok 4.3 as a new planning generator.
GPT-5.4 Mini independently generates all hidden targets, validation supports,
and endpoint supports. Both models are nonreasoning.

The run is authorized only if every frozen Grok exact-10 serving gate passes.
It is distinct from the closed DeepSeek and Qwen routes: new planner, fresh
seeds, separate result directory, and no reused responses.

## Frozen Design

- 32 trees, planning seeds `45000..45031`;
- target seeds `45100..45131`;
- eight validation draws per tree beginning at `45200`;
- 15 extra endpoint draws per tree beginning at `45500`, plus the target
  draw, for 16 endpoint draws per tree;
- 49 Grok planning calls and 24 GPT-5.4 Mini endpoint-family calls per tree;
- exactly 2,336 accepted responses;
- temperature `0.7`, explicit nonreasoning mode;
- retained-parent rejuvenation at both refreshes;
- eight independent validation supports select both depth-three and
  depth-two roots;
- the same 16 independent endpoint supports evaluate every policy;
- maximum reported cost and minimum starting balance `$7.50`.

All policy mechanics, parsers, filters, candidate roots, PTS control,
cross-fitted scoring, endpoint aggregation, and bootstrap code are unchanged.

## Confirmatory Gates

Every gate must pass:

1. Exact request accounting, at most eight retries, zero reasoning tokens,
   zero forced exits, and cost within budget.
2. Initial supports have at least 16 valid rules; validation and endpoint
   supports at least 16; retained first branches at least eight; retained
   second branches at least four.
3. Every tree has at least 128 endpoint hypotheses novel to its initial
   support.
4. Depth three and equally cross-fitted depth two choose different roots on
   at least 12 trees.
5. Depth three improves endpoint posterior-predictive Brier by at least 1%,
   the whole-tree bootstrap difference interval lies below zero, and it wins
   at least 12 trees.
6. Novel-target Brier does not regress.
7. Depth three has a negative Brier difference with an interval below zero
   against myopic EIG, fixed-support depth three, and PTS.
8. Mean depth-three source-to-endpoint Brier Spearman is at least `0.7` and
   exceeds depth two by at least `0.15`.

Brier is confirmatory because it is the policy objective. Hamming and exact
coverage are reported as secondary diagnostics, not folded into a
post-selection conjunction. Any parse failure, missing tree, response
replacement, selective removal, or post-response threshold change fails the
run closed.
