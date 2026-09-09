# Reuse the later predictive-risk path, not the truth-controlled entropy scorer

The previous audit concerned generator_aware_score only. The later
predictive_bayes_risk_scores in the same hash-bound source delegates directly to
evaluate_policy_root, passing initial hypotheses as simulated target worlds.
That evaluator selects the continuation from the generated branch alone. Truth
supplies observed answers and evaluation losses, not inserted belief hypotheses.

A new regression uses the same indistinguishable-root fixture as the historical
failure. Both hidden worlds now select query 1. Branch size remains one; the
branch list is unchanged. The inconsistent second answer yields zero survivors,
not an injected-truth repair. Evaluating the same targets through the execution
entrypoint yields identical per-target rows and mean Brier. Combined source and
boundary tests: 12 passed in 0.94 seconds. No banked endpoint was rerun.

This is a useful negative check on the previous diagnosis: do NOT attribute later
predictive-risk results to the entropy scorer's truth-injection problem. No
replacement branch engine is justified by that finding alone. The existing
predictive evaluator is the reuse candidate for a future qualified interface.

Remaining limitations are distinct: it uses EIG for the continuation, its outer
world distribution is the initial support, and terminal scoring excludes queried
coordinates. A future same-objective depth comparison must freeze identical
target coordinates and experiment budgets, and optimize the same predictive
objective at each horizon. Its generated hypotheses still need predictive-transfer
qualification. This fixture proves neither calibration nor a positive depth gain.

Decision: do not spend on replaying old Number Game generations or rewriting
already-correct truth isolation. Any new experiment must qualify a source-backed
history-conditioned hypothesis generator and its outer predictive distribution
first. The latest coverage reversal, not the old entropy bug, remains the primary
empirical obstacle for the corrected predictor.

Previous and current turns are progress; current work narrows a potential causal
overstatement and protects a reusable implementation. Cost $0. Authenticated
account usage 221.306531939, balance 23.693468061, remaining conservative daily
allowance 4.11174654 unchanged. Goal active/unachieved. No cluster or automation
changes; no paid endpoint authorized.
