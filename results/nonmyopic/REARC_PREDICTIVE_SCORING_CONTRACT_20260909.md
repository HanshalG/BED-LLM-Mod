# Shared predictive scores for variable-size grid outcomes

Implemented scripts/rearc_predictive_score.py before any RE-ARC model proposals.
It accepts weighted complete-grid predictions and scores the resulting mixture,
not its best member. Duplicate predicted grids aggregate probability; they are
not deduplicated/reweighted. Supplied weights must already sum to one.

Whole-grid categorical Brier is (1-2p(truth)+sum p(grid)^2)/2, in[0,1]. Its
outcome space includes every allowed grid and a distinct failed-execution outcome.
Wrong shapes are ordinary incorrect outcomes, not discarded cases. Whole-grid
log loss is also computed; zero truth probability remains infinite, never clipped
to a favorable finite number. A future strict JSON writer must encode that infinity
explicitly, e.g. null plus a nonfinite-status flag, not an invalid JSON token.

The supplementary fixed-canvas score embeds grids in a30x30canvas, with colors0..9,
padding10 and failure11. It averages normalized categorical Brier across the SAME
900coordinates for every policy and depth. This is a proper marginal score, not a
strictly proper score for the full joint distribution: correlations matter for
rollouts and are retained through whole-program/world sampling. Padding may dominate
small examples, so do not use this metric alone to claim rule discovery. Whole-grid
scoring may be insensitive when every proposed rule is slightly wrong; report both.

None or malformed grid outputs retain their supplied mass as execution failures.
All failed output mass incurs unit loss on both scores. An empty forecast raises
an explicit failure; future experiment aggregation must retain that case rather
than select a successful subset. Scoring does not infer Bayesian program weights,
choose a prior, prune programs, or decide how to update after demonstrations.

Seven tests pass in .10s: probability aggregation, failure retention, variable-shape
handling, zero-probability log loss, invalid-weight rejection, and a proper-score
identity (expected excess Brier equals squared probability error in a binary case).
No benchmark outputs or paid responses were opened for these tests.

Next freeze the actual proposal qualification, including its predictive mixture
weighting/invalid-case handling and symbolic baseline, before generating the example
panel or sending Luna-medium requests. Scores are reusable mechanics, not a passed
semantic gate or proof of non-myopic headroom. Historical outcomes remain untouched.

Previous turn was source progress; current turn makes the common predictive outcome
contract executable. Cost$0, balance23.693468061, remaining4.11174654 unchanged.
Goal active/unachieved; no cluster or automation changes.
