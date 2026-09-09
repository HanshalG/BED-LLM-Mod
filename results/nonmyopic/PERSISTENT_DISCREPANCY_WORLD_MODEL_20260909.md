# Candidate persistent-discrepancy model for out-of-support worlds

## Source-informed decision

[Maximin Robust Bayesian Experimental Design, v1](https://arxiv.org/html/2603.14094v1)
jointly robustifies the data-generating distribution and posterior update, rather
than tempering inference alone. Its KL ambiguity set uses KL(q || p), and its
posterior has likelihood raised to a positive alpha. In our finite deterministic
setting, a finite KL radius cannot give positive mass where p is zero; likewise
0^alpha=0 and 1^alpha=1. This is our application of the formulas, not an experimental
claim from that paper. Merely adding a temperature would not fix missing rules.

The OpenReview robustness paper encountered a browser verification page; its full
text was not reviewed. A recent Nature uncertainty-calibrated-optimizer paper was
located but not used to justify this implementation. No claim of a complete new
literature review or source-qualified environment follows.

## Implemented candidate

scripts/number_game_persistent_discrepancy.py defines an opt-in generative model:

1. Choose a unique executable rule uniformly from the proposed pool.
2. Draw an error rate epsilon from a declared Beta(alpha,beta) prior.
3. Draw a persistent discrepancy bit for each coordinate independently given
   epsilon. The complete hidden concept is rule XOR those bits.
4. Observations reveal bits of that SAME concept, without measurement noise.

For each rule, e disagreements in n unique observed coordinates give conditional
Beta(alpha+e,beta+n-e) and marginal likelihood B(alpha+e,beta+n-e)/B(alpha,beta).
Normalize these evidences for rule weights. Observed coordinates have exact known
values; unobserved predictions average the conditional flip probability across
rules. Sampling returns an entire persistent world, including observed values.
Callers must sample once per simulated trajectory, not redraw at each action.

Positive hyperparameters give support to every Boolean extension. They are required
arguments, not defaults fitted to opened transport losses. Duplicate observations
do not count twice; contradictory labels fail because the source is deterministic.
The public observation updater never receives a simulated truth object.

Four tests pass in .08s: exact enumeration of every three-bit world reproduces
normalization and conditional predictions; repeated answers do not add evidence;
sampled worlds retain observed answers; predictive tower identity and expected
one-query Brier are coherent; deterministic tempering cannot revive zero mass.

## What this does not establish

Full support is not semantic calibration. The exchangeable error model may be a
poor approximation to structured missing rules and may reduce useful information.
It does not generate explanations or replace the LLM's discovery role. A
discrepancy-only model and equal-call blind LLM pools are mandatory productive
controls. A sufficiently flexible discrepancy layer could make the LLM ornamental.

This changes no historical runner or endpoint. No hyperparameters were searched,
no old donor predictions were refitted, and no positive depth result is claimed.
The 1/3 and 1/4 Beta choices in tests are arithmetic fixtures only. A scientific
study must bind hyperparameters prospectively or learn them on an independent
development set, qualify prequential predictions on fresh source-backed worlds,
and check simulator/real-update fidelity before policy comparison. The long-term
target remains LLM-generated, answer-conditioned structure discovery with a real
same-objective non-myopic advantage, not this small classical model.

Next: determine whether structured residual models or this simple candidate can
qualify predictive transfer with independently justified settings, before spending
on a new depth grid. Do not interpret exact coherence as empirical calibration.

Previous turn was progress; current turn implements a coherent outer world model
that addresses zero support and persistent-answer consistency. Cost $0, balance
23.693468061, conservative daily remaining4.11174654 unchanged. Goal unachieved.
