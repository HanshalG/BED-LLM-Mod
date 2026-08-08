# Bongard OpenWorld LLM-Native Computational-Role Amendment

Frozen: 2026-08-08, before any Bongard mechanics, development, or confirmation
model response and before any candidate or scientific endpoint label was
opened.

This amendment narrows the manuscript claim to the computation that is
actually implemented. It changes no model, prompt, response schema, task,
seed, action, endpoint, policy, score, gate, request count, or budget.

## Exact Computational Role

At each root or labelled history, Luna emits ten predictive particles. Each
particle contains:

- a free-form visual rule for semantic interpretation and uniqueness checks;
- a history-conditioned particle weight;
- one positive-label probability for each of the 14 opaque images.

After strict parsing, deterministic BED utilities consume the normalized
weights and the 10-by-14 probability matrix. They do not numerically inspect
the free-form rule strings. Replacing every rule string while preserving the
weights and probabilities therefore leaves one-step endpoint PIG,
fixed-support depth-two scores, path-dependent depth-two scores, selected
actions, and endpoint predictions unchanged. A dedicated invariance test binds
this fact before outcomes.

The irreducible LLM contribution is consequently not arithmetic over rule
names. It is the history-conditioned construction of the predictive particles
and their counterfactual transition matrices from pixels and labelled history.
Once those matrices exist, the Bayesian update and lookahead are classical and
exact. The fixed-support and history-blind controls test whether reusing or
regenerating those LLM-produced matrices differently changes planning and
endpoint prediction.

## Claim Boundary

The paper may call the design multimodal and LLM-native because every root and
branch predictive particle matrix is supplied by Luna from an open-ended
visual concept task. It must also state or preserve these boundaries:

- rule strings are interpretive particle descriptions, not numerical planner
  inputs;
- model-emitted probabilities are called predictive probabilities or
  likelihood estimates, not calibrated likelihoods before calibration is
  empirically established;
- a positive result establishes value over the registered fixed-support,
  history-blind, matched-updater, shuffled, random, and myopic controls;
- it does not prove that every possible classical vision system or separately
  trained likelihood model is incapable of supplying an alternative matrix.

This narrower wording strengthens rather than changes the registered estimand:
non-myopic value is attributed to answer-conditioned LLM predictive-belief
dynamics only when the complete frozen conjunction passes.
