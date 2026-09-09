# Objective-aligned continuation primitive

Added an opt-in, I/O-free continuation implementation in
scripts/number_game_fixed_target_risk.py. Historical runners and results are
unchanged. This is implementation progress, not an executed new policy study.

The selector minimizes exact expected terminal Brier on an explicit fixed target
tuple. Query candidates are separate from targets: observing a query never removes
its coordinate from the scoring denominator. Uniform-over-unique-extension prior
is explicit; duplicate extensions are rejected, not silently reweighted.

The public continuation takes only hypotheses, history, candidate queries, and
target indices. It filters using full history, excludes used queries, and has no
truth parameter. Truth exists only in terminal_brier, the evaluator. Empty support
raises an explicit error rather than resetting the prior or injecting a world.
Future runners must retain such failures in their accounting rather than omit them.

Tests verify exact agreement of expected loss with explicit persistent-world
enumeration, a target-relevant query chosen over an entropy-tied irrelevant query,
support-order invariance, fixed target denominator, public-history filtering, and
failure behavior. Seven focused tests including the information-boundary suite
pass in 0.17s. No model calls or banked policy endpoints were executed.

This primitive is not a complete dynamic horizon planner. It supplies the corrected
terminal one-step policy for a prospective runner; arbitrary branch refresh still
needs a qualified generator and transition model. It deliberately does not claim
that a longer horizon is monotonically better under misspecified beliefs. The
initial-support transport failure remains unresolved and the original gates remain
closed. Do not retrofit old result numbers with this changed objective.

Next integration requires a fresh source-backed predictive qualification with
history-conditioned and equal-call history-blind proposals, then frozen same-target
root planning using this selector in both simulation and execution. Reuse existing
branch evaluation structure; do not build another independent simulator.

Previous turn was progress; current turn implements the objective-aligned selector.
Cost $0; authenticated usage 221.306531939 and balance 23.693468061 unchanged.
Remaining conservative allowance 4.11174654. Goal active/unachieved.
