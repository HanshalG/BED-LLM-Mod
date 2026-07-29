# Number Game Gemini Classical-Grammar Audit Preregistration

Status: **frozen before classifying any Gemini endpoint against the grammar
bank or scoring the grammar-novel subset**.

Date: 2026-07-29

Cost: zero model calls and zero OpenRouter spend.

## Question

The prior classical-grammar audit showed that 50.98% of unique generated
second-refresh extensions lie outside a deterministic bank of 416,366
classical Number Game concepts, but Qwen endpoints supplied too few
out-of-bank targets to evaluate the fixed policies.

This retrospective audit asks:

> Across the two independent 32-tree Gemini endpoint studies that establish
> the depth-three result, are there enough out-of-bank targets, and does
> fixed-root depth three still beat equally cross-fitted depth two on them?

No policy is regenerated or reselected. This is a preregistered retrospective
subset audit of already-open endpoint artifacts, not a fresh confirmation.

## Frozen Grammar And Support Evidence

Reuse the exact extension-level grammar definition in
`NUMBER_GAME_CLASSICAL_GRAMMAR_IRREDUCIBILITY_PREREGISTRATION.md`.
The rebuilt bank must:

- contain at least 100,000 unique nonconstant extensions; and
- have canonical SHA-256
  `6e2a523d7d7c0c59b5df0494de37087525c7009d43a5eaba01390e005d85e0e0`.

Bind the prior audit result SHA-256
`514945d268712227c128ecb9be9ee0a495cf1682a1c53c99ff5428b8da0bf9a7`.
Its generated-support novelty gates must remain true. No grammar atom,
closure, parameter, or threshold may change.

## Frozen Evidence

Study 1, fixed policies with fresh high-precision endpoints:

- source trees:
  `cf239683033be3fbfed6a449f2aaa1ad1bcbe57d935ca21cf43e46ba938ea7f9`
- source result:
  `47e684b11ac5c6340ff4f063ff7a14db8b8d0bd18b184d61f91e315903e4809e`
- 16-draw Gemini endpoints:
  `0e6bd789b28a77f2ca47a13015fc80d46686daf14f3a2540caebe8663eff17c0`

Study 2, wholly fresh trees and endpoints:

- source trees:
  `197dcfe3cb48eeb9a4b656d0f9b20ef2e0b45ac8d1df0ec4bd9358e07af18802`
- source result:
  `25e0939164f6806b30481e48a88372f22639f6065096cb59978600509d43a3d8`
- 16-draw Gemini endpoints:
  `103626763fe25987541a48adb93ac1dea600f53ecf64b95d1d4dbc2d66efc824`

This gives 64 fixed trees and 1,024 independent Gemini endpoint-support draws.
Tree order and existing selected roots are immutable.

## Frozen Power Gate

Classify every endpoint hypothesis by exact 101-bit extension membership in
the bank. A draw is nonempty if it contains at least one grammar-novel target.

All are required:

- at least 5% of endpoint occurrences are grammar-novel;
- at least 512 grammar-novel target occurrences overall;
- at least 48 of 64 trees have at least 8 of 16 nonempty draws; and
- each 32-tree study contributes at least 24 analyzable trees.

If this gate fails, status is `inconclusive` and no efficacy claim or paid
follow-up is authorized.

## Frozen Fixed-Policy Efficacy

For each nonempty draw, evaluate the already selected cross-fitted
depth-three and depth-two roots on only grammar-novel targets. Retained branch
supports and execution are unchanged. Average draws within tree; trees are
independent units. Include only trees meeting the eight-draw rule.

All are required:

- mean depth-three Brier at least 1% below depth two;
- 20,000-sample tree-bootstrap 95% interval for paired
  `depth_three - depth_two` Brier has upper endpoint below zero;
- at least 20 tree wins;
- mean Hamming does not increase;
- mean exact-extension coverage does not decrease; and
- mean Brier difference is negative in both source studies.

Bootstrap seed: `37291`.

## Interpretation And Follow-Up

- `positive_irreducibility_audit`: bank, power, and every efficacy gate pass.
- `negative`: endpoint power passes but efficacy fails.
- `inconclusive`: endpoint power fails.
- `mechanics_failure`: source or bank binding fails.

There is no grammar narrowing, target deletion, lower analyzability threshold,
root reselection, alternate metric, or selective source reporting after the
audit opens.

Only `positive_irreducibility_audit` authorizes a separately preregistered
fresh endpoint-only confirmation on the same 64 fixed policies. Any other
status closes this route without paid calls.
