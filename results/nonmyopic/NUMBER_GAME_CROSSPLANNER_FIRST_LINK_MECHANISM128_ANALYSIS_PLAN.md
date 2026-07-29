# Number Game Cross-Planner First-Link Mechanism-128 Analysis Plan

Date frozen: 2026-07-29, after all four source outcomes were open and before
the 128-tree mechanism aggregate was computed.

This is retrospective mechanism analysis. It cannot rescue, replace, or change
the registered status of any source study.

## Question

Across both Qwen 3.7 Plus and GPT-5.4 Mini planning generators, does the
simulated-risk advantage of the path-dependent depth-three root predict its
exact-canonical realized Brier advantage over a competing root?

This tests only the first link from simulated policy value to endpoint utility.
It does not introduce online observation or posterior-execution noise.

## Fixed Sources

- two disjoint 32-tree Qwen blocks already bound by the pooled-64 result;
- two disjoint 32-tree GPT-5.4 Mini blocks already bound by the canonical
  replay-64 result;
- 33 exact Tenenbaum--Griffiths endpoint concepts per tree;
- all 128 planning-tree seeds disjoint.

## Fixed Analysis

For myopic EIG, fixed-support depth three, and cross-fitted depth two:

1. record whether the competing root differs from path-dependent depth three;
2. compute predicted advantage as the cross-fitted depth-three simulated Brier
   risk of the competing root minus that of the selected root;
3. compute realized advantage as exact-canonical per-root Brier for the
   competing root minus that of the selected root;
4. report means, wins/ties/losses, and Spearman correlation on all trees and
   changed-root trees;
5. report each 32-tree block, each planner family, and all 128 trees.

Use 20,000 bootstrap samples, resampling 32 trees independently inside every
source block. Family summaries preserve their two constituent blocks.

There are no pass thresholds. Positive means and correlations are descriptive
mechanism evidence; negative or heterogeneous results are reported unchanged.
Depth-three versus depth-two remains a boundary diagnostic, not a monotonic
depth claim.

Model calls and cost are zero.
