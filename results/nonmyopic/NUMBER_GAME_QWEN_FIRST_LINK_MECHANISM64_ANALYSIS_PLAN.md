# Number Game Qwen First-Link Mechanism-64 Analysis Plan

Date frozen: 2026-07-29, after both source cohorts and their pooled endpoint
were open. This is retrospective mechanism analysis, not a new efficacy test.

## Question

When path-dependent depth-three planning selects a different root, does its
simulated-risk advantage over the competing root predict the exact-canonical
realized Brier advantage?

This isolates the first link between simulator ranking and endpoint utility.
It does not introduce realized online observation noise or test a second
execution link.

## Fixed Analysis

- Hash-bind the same two independent 32-tree Qwen canonical cohorts.
- For myopic EIG, fixed-support depth three, and cross-fitted depth two:
  - record whether the root differs from path-dependent depth three;
  - compute predicted advantage as the cross-fitted depth-three risk of the
    baseline root minus that of the selected root;
  - compute realized advantage from exact-canonical per-root Brier in the
    same tree;
  - report mean advantage, wins/ties/losses, and Spearman correlation on all
    trees and the changed-root subset.
- Use 20,000 stratified bootstrap samples, resampling 32 trees independently
  within each source cohort.

No threshold can pass or fail, no source status changes, and no model call is
made.
