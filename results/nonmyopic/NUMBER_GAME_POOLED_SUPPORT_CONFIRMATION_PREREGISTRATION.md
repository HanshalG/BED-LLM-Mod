# Number Game Pooled-Support Confirmation Preregistration

Date frozen: 2026-07-28, before any confirmation response.

The post-hoc pooled-support audit is fully exposed. This study confirms on
fresh trees whether root-conditioned LLM generation beats a support-rich
classical depth-two control.

## Frozen Design

- 32 fresh Gemini 2.5 Flash planning trees, seeds `27200..27231`.
- 32 fresh GPT-5.4 target supports, seeds `27300..27331`.
- Nonreasoning, temperature `0.7`, strict 24-rule schema, eight candidate
  roots, both answer branches, terminal predictive Bayes risk, and greedy-EIG
  second query are unchanged.
- The new control deduplicates all rules generated across all roots and
  answers into one static support per tree. It filters this common support by
  each simulated answer and selects a root by the same exact terminal-risk
  objective. Endpoints use the unchanged realized branch supports.
- The root-local pooled identity is also audited: pooling the two answer
  supports for a root and filtering by label must recover the
  generator-aware selection on every tree.
- Exactly 576 accepted responses. Explicit zero-cost provider errors may
  receive only the existing identical-payload transport retry.
- Equal tree weights and 50,000 whole-tree bootstrap samples. No development
  tree is pooled.
- Total cost cap: `$3.00`; no reserve.

## Structural Gates

All trees must retain at least 16 initial rules, eight rules in every branch,
16 target rules, and eight target extensions novel to planning support.
Attempt accounting must be exact, with zero reasoning tokens and forced exits.

## Scientific Pass Criteria

All must pass:

1. Root-local pooled identity holds on 32/32 trees.
2. Root-conditioned and global-pool roots differ on at least 20/32 trees.
3. Versus global pooled support: at least 8% aggregate Brier gain, wholly
   negative whole-tree Brier interval, and at least 18/32 Brier wins.
4. At least 10% Hamming reduction, wholly negative Hamming interval, and at
   least 16/32 Hamming wins.
5. No mean exact-extension coverage loss.
6. Extension-novel targets have negative mean Brier and Hamming differences.

Passing isolates root-conditioned LLM proposal dynamics from static support
quantity. It does not show that answer attribution matters after all
counterfactual rules have already been generated, nor that the restricted
Number Game grammar is representative of unrestricted natural-language BED.
