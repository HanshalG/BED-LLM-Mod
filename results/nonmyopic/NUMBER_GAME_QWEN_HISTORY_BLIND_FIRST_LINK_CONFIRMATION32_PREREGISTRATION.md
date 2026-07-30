# Number Game Qwen History-Blind First-Link Confirmation-32 Preregistration

Date frozen: 2026-07-30

## Purpose

This is a fresh confirmatory test of the strongest mechanism signal discovered
in the matched history-blind V3 development cohort. It asks whether the
root-specific benefit of answer-conditioned support predicts the realized
advantage of the dynamic-support planner over the fixed-support planner.

The V3 selected-root mean-contrast gate remains failed. This confirmation
cannot rescue or relabel V3, the 96-tree source study, or any earlier source
study.

## Immutable Sources

The source study is the fresh resilient 96-tree Qwen study:

- source `RESULT.json` SHA256:
  `04177da415498d765baa2b460f6d732a5162b118776df4d5019f7b540bfce36e`;
- source `TREES.json` SHA256:
  `8535df7eba5437c564cc6df56816fcee0a6991c3358fdaa6b62c6ca96948da82`;
- source `TARGETS.json` SHA256:
  `2fd09b75bcce734e17f237ced9a49e97e13e290e2f5229b9354ad7a8a1bdafd`.

The development result is matched history-blind V3:

- V3 `RESULT.json` SHA256:
  `29bb76074dfb53a6efef352fde88fbca6c3a7dc591d0936d132c82050f1dc71d`;
- V3 `CONTROLS.json` SHA256:
  `6db0e3e272cb843250174a2252098c4a88d3d4a154d3287ba5542f9d2eb09a1b`.

V3 used source tree indices `0..31`. This confirmation uses the next
contiguous, disjoint source block, indices `32..63`, with seeds
`80032..80063`. The block is fixed without optimizing on the source outcomes.
It contains 21 trees on which the stored dynamic and fixed planners selected
different roots, clearing the minimum-20 structural gate before any new
control response is generated.

The source outcomes are pre-existing and public. The new root-specific
history-blind control values, and therefore the registered correlation
endpoint, are unopened at freeze time.

## Model And Requests

- model: `qwen/qwen3.7-plus`;
- reasoning: disabled;
- prompt: the same initial Number Game prompt with no observations used by
  V3;
- parser, validity filtering, consistency filtering, and recursive parent
  retention: exactly the V3 implementation;
- 32 trees, 48 branch slots per tree, and two independent draws per slot;
- exactly 3,072 accepted requests and zero target/validator requests;
- fresh control seeds begin at `9,600,000`, using
  `9,600,000 + 1000 * local_tree_index + 2 * slot_index + draw_index`;
- concurrency: 256;
- run-cost cap: `$4.25`;
- minimum authenticated starting balance: `$5.00`;
- bootstrap seed: `9,700,000`;
- bootstrap samples: 20,000.

No V1, V2, or V3 response may be reused.

## Mechanics Gates

The endpoint remains inaccessible unless all of these pass:

1. the exact source block, tree seeds, 1,536 branch slots, and 3,072 accepted
   requests are present;
2. HTTP-attempt accounting is exact, with at most 32 retries and at most 32
   provider-error retries;
3. every draw is strict JSON and contains at least 16 valid unique rules;
4. every two-draw pool contains at least 24 valid unique rules;
5. reasoning tokens and forced exits are both zero;
6. total run cost is no more than `$4.25`.

As in V3, second-draw novelty is descriptive and is not a gate.

## Scoring

For every source root and every canonical target path, recompute the
first- and second-stage posterior-predictive MSE, truth-extension coverage, and
support size for:

- the stored answer-conditioned support; and
- a fresh history-blind support generated from the initial prompt, followed
  by the same filtering and parent retention.

For each changed-root tree define:

- conditioning benefit at a root =
  `history-blind second-stage MSE - conditional second-stage MSE`;
- conditioning-benefit contrast =
  `benefit(dynamic-selected root) - benefit(fixed-selected root)`;
- realized advantage =
  `realized Brier(fixed-selected root) - realized Brier(dynamic-selected root)`.

The primary first-link endpoint is Spearman correlation across changed-root
trees between conditioning-benefit contrast and realized advantage. Its 95%
interval uses a paired tree bootstrap over changed-root trees.

## Scientific Gates

The confirmation passes only if all four gates pass:

1. at least 20 source trees have different dynamic and fixed roots;
2. the 95% interval for all-root second-stage
   `conditional MSE - history-blind MSE` is entirely below zero;
3. the 95% interval for all-root second-stage
   `conditional coverage - history-blind coverage` has lower endpoint at
   least zero;
4. the 95% interval for the changed-root first-link Spearman correlation is
   entirely above zero.

The selected-root mean conditioning-benefit contrast is reported
descriptively, including its interval, but is not retested as a success gate.

## Decision And Failure Rules

- `passed`: every mechanics and scientific gate passes;
- `gated_null`: mechanics pass but at least one scientific gate fails;
- `mechanics_failed`: any mechanics gate fails, with no scientific endpoint
  emitted;
- `failed_closed`: transport or evaluator execution fails.

There is one paid execution. No subset rescue, seed substitution, response
reuse, endpoint-driven rerun, or change to the source-study status is allowed.
An independent zero-call replay from the saved public controls must reproduce
the complete endpoint before the result is banked.
