# Number Game Qwen Pooled Second-Refresh Confirmation-32 Preregistration

Date frozen: 2026-07-29, after the retrospective second-refresh ablation
passed and before any seed in `64000..64501` was sent to a model.

## Fixed Fresh Study

- 32 fresh trees, seeds `64000..64031`.
- Qwen 3.7 Plus nonreasoning planning with two independent generations at
  every one of 49 planning histories; second seed offset `1,000,000`.
- Gemini 2.5 Flash target seeds `64100..64131`.
- Eight Gemini cross-fit validation supports per tree, seeds
  `64200..64455`.
- Unchanged retained rejuvenation and exact 33-concept canonical endpoint.
- 20,000 paired bootstrap samples, seed `64500`; first-link diagnostic seed
  `64501`.

The exact-10 pooled serving smoke and completed prior 32-tree run validate the
unchanged model, prompts, parser, pooling adapter, and support update. This
runner changes only the preregistered endpoint analysis, so no new serving
smoke is required.

## Primary Mechanism

For every tree, cross-fit depth-three root selection is recomputed using the
same retained first support, second queries, and eight validation draws with:

1. merged retained-plus-new second-step support;
2. parent-only second-step support.

The mechanism passes only if roots differ on at least 12 trees, merged support
reduces exact Brier by at least 2%, the paired tree-bootstrap difference
interval is below zero, and wins minus losses is at least eight.

Generated-only second support is diagnostic.

## Required Policy Efficacy

The merged-support depth-three policy must also beat myopic EIG by at least 8%
exact Brier, have its paired tree-bootstrap interval below zero, and win at
least 20 trees. Both mechanism and policy gates are required.

## Mechanics

- Exactly 3,424 accepted requests and exact attempt accounting.
- At most 96 transparent retries/provider-error retries.
- Zero reasoning tokens and forced exits; at most 16 item-salvaged draws.
- Cost at most `$5.25`; starting provider-visible balance at least `$5.50`.
- Every pooled initial support has at least 24 valid extensions.
- Every deployed retained first/second support has at least 12/eight.
- All eight validation supports per tree have at least 16.

Generated-only support minima are reported but are not execution gates because
generated-only is an ablation, not the deployed belief state.

The result is reported once as `passed`, `gated_null`, or `failed_closed`.
There is no continuation, seed replacement, third draw, semantic repair,
threshold change, or endpoint change. Prior statuses remain unchanged.
