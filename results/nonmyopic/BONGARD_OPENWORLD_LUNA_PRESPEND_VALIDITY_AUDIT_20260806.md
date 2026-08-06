# Bongard Luna Pre-Spend Validity Audit

Date: 2026-08-06

## Outcome

The primary `dynamic_depth2` and `myopic_width` policies remain valid and
unchanged. The pre-spend audit found one invalid negative-control
implementation and corrected it before any Bongard model request or endpoint
access.

## Finding

The old shuffled control rotated regenerated branch-support objects from a
source first action to a different target first action. The generic dynamic
scorer then excluded the target action from the second-query candidates, even
though the branch history had observed the source action. This simultaneously:

- left the already-observed source image eligible for a second query; and
- removed the still-unobserved target image.

It therefore did not preserve valid query eligibility or the quality
distribution of complete continuation values. A comparison against this
control could have overstated the importance of correctly aligned LLM belief
dynamics.

## Correction

The corrected control computes each real action's complete expected
continuation value under its own branch supports and legal remaining-query
set, then cyclically permutes those scalar values across root actions. This
preserves their multiset exactly and breaks only the action-to-future-value
coupling. Execution after selection still uses the selected action's real
branch.

Every nonrandom policy now also reports its selected-score margin. A
dynamic-versus-myopic action change counts toward a gate only when its dynamic
advantage clears `1e-6` nats, excluding numerical ties.

## Verification

- full Bongard test suite: `58 passed`;
- synthetic four-task full tree still passes every mechanics gate;
- counterfactual test flips every unreleased candidate label and confirms that
  all root scores and first actions remain unchanged;
- shuffled continuation values are checked as an exact permutation and the
  control function accepts no branch-support map;
- development protocol manifest independently verifies all 32 opaque tasks,
  all gates, and nine implementation/protocol hashes;
- corrected development interface: `-3`;
- corrected manifest SHA-256:
  `d5e8412f6e2f485a357ba255692f1c6d60a99b4900e588b05ad39b9f276b5b9c`.

No model request, scientific endpoint, confirmation task, or sealed-test task
was opened by this audit.
