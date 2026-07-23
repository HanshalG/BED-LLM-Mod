# UCI Thyroid Workup 26B Trajectory Confirmation Preregistration

Status: frozen after the named-proposal S1 gate passed and before any response or
endpoint from the trajectory confirmation is observed.

## Design

- Fresh seed `24156`; 50 patient rows sampled without replacement from the complete
  7,200-row UCI ann-thyroid empirical prior.
- Eight paired actions per patient across four arms: the non-thinking Gemma 4 26B
  named-continuation policy, matched-random named continuations, exact depth one,
  and exhaustive exact depth two.
- On rounds 1--7, the LLM supplies one named second action for every branch of four
  machine-fixed roots. The exact verifier scores the four complete two-action
  policies and executes the selected root. On round 8, every arm uses its registered
  exact one-step decision because no second action remains in the endpoint horizon.
- The matched-random arm uses identical machine-root construction on its own state,
  random named legal continuations from a deterministic trial/round seed, and the
  same exact verifier.
- The exact depth-one and depth-two controls use the same finite-population posterior,
  observations, target entropy, and tie-breaking convention.
- Ten thousand paired bootstrap replicates. Primary performance is total eight-round
  mean post-action entropy AUC; truth-log probability AUC is the truth-anchored
  corroboration. All arms use the same hidden patient in each paired trial.

## Frozen gates

All are required:

1. LLM entropy-AUC gain over exact depth one has a positive paired 95% lower bound.
2. LLM truth-log-AUC gain over exact depth one has a positive paired 95% lower bound.
3. LLM entropy-AUC gain over matched random has a positive paired 95% lower bound.
4. LLM truth-log-AUC gain over matched random has a positive paired 95% lower bound.
5. Mean LLM entropy-AUC gain over exact depth one is at least 60% of the mean
   exhaustive-depth-two gain over exact depth one.
6. The LLM selects blood collection first on at least 75% of trajectories.
7. Fifty distinct paired truths and all four 8-round traces complete; all actions are
   legal; exactly 350 logical LLM cells are accepted; reasoning tokens, forced exits,
   and rollout/scoring-time LLM calls are zero.

The one registered validation-feedback retry remains available for malformed named
JSON. Failure after that retry stops the confirmation. No parser repair, threshold
change, alternate seed, or replacement trial is allowed after endpoints are viewed.

Model and serving: `google/gemma-4-26B-A4B-it`, direct vLLM on one A100 in `msc,llm`,
non-thinking, temperature zero, 1,024-token cap. No OpenRouter spend is authorized.

A pass requires an independent full trace/control replay before the result is banked.
A failure is reported as the transfer boundary and does not invalidate the exact
qualification or the passed proposal-quality gate.
