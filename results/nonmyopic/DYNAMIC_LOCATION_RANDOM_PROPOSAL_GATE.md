# Dynamic Location Random-Proposal Gate

**Status:** zero-cost non-start. No OpenRouter or other LLM calls were made for
this gate, and the preregistered dynamic-location LLM pilot was not launched.

## Purpose

The locality-constrained branch-decoy/local-bump environment has a strong
full-action exact-oracle depth-two gap. Before paying for an LLM candidate proposer,
we tested whether that gap survives the study's required compute-matched one-step
width control when candidates are restricted. The control replaces the LLM with a
deterministic, prompt-conditioned random legal proposer. It retains the exact
likelihood, posterior, EIG scorer, local-action constraint, common random numbers,
candidate cache, and d2 call allocation from the preregistered harness.

The comparison is therefore a proposal-coverage gate, not LLM evidence.

## Reproducible Harness

The control is implemented by `PromptRandomLegalCandidateModel` in
`scripts/nonmyopic_dynamic_location_pilot.py`. It samples two distinct legal action
IDs from each prompt, preferring not to repeat earlier current-state width IDs. The
same source, initial particles, and observation noise are shared across all three
arms. `d1_matched_width` receives the same number of candidate calls as d2's root
plus its counterfactual future cells.

Example cohort command:

```bash
PYTHONPATH=. python scripts/nonmyopic_dynamic_location_pilot.py \
  --random-legal --num-trials 10 --num-rounds 6 --seed 1304 \
  --bootstrap-replicates 10000 \
  --output-dir /tmp/dynamic_location_random_grid11_seed1304
```

## Results

The default 11x11 geometry has five legal moves after the fixed origin query. d2
uses 13 candidate cells per nonterminal decision; thus the width arm effectively
exhausts the local action set. Across ten independent 10-trial cohorts (seeds
1304--1313; 100 paired trajectories), d2 improves over its shared K=2 one-step
baseline but not over matched width:

| Comparison | d2 minus control RMSE AUC | Descriptive bootstrap 95% CI | W / T / L |
| --- | ---: | --- | --- |
| d2 - shared d1 | -0.0265 | [-0.0513, -0.0070] | 17 / 71 / 12 |
| d2 - matched width | +0.0184 | [-0.0295, +0.0669] | 36 / 1 / 63 |

At 31x31, there are 29 local legal actions, so the width control no longer
trivially enumerates the action set. A five-trial exact-oracle check still had an
early RMSE-AUC advantage for depth two, but essentially no final-RMSE advantage.
More importantly, three independent 10-trial random-proposal cohorts (seeds
1304--1306; 30 paired trajectories) rejected the restricted-candidate transfer:

| Comparison | d2 minus control RMSE AUC | Descriptive bootstrap 95% CI | W / T / L |
| --- | ---: | --- | --- |
| d2 - shared d1 | -0.0102 | [-0.0312, +0.0008] | 3 / 25 / 2 |
| d2 - matched width | +0.2407 | [+0.1457, +0.3364] | 6 / 0 / 24 |

Negative values favor d2. The intervals are only descriptive diagnostics; the
decision does not rely on inferential claims.

## Decision

This geometry does not meet the depth-versus-compute-matched-width gate for a paid
LLM pilot. The original 11x11 setting gives the width control too little distinct
action space, while the finer 31x31 setting makes the restricted d2 root proposal
much weaker than current-state coverage. Do not launch
`config_nonmyopic_dynamic_location_pilot_openrouter.yaml` under this preregistration.
The exact full-action oracle remains a useful correctness control but is not evidence
that a restricted LLM candidate planner has the required depth advantage.
