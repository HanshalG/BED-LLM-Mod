# Mushroom Feature Acquisition 26B Serving Smoke Preregistration

Registered 2026-07-23 after the exact depth qualification passed and before any
Mushroom prompt was sent to an LLM.

## Purpose

Validate the strict indexed semantic-policy interface before measuring proposal
quality. S0 is a serving and mechanics gate only. Its policy scores, selected feature
quality, and comparison endpoints are not qualification evidence and will not be used
to tune this interface.

## Frozen S0

- Model: `google/gemma-4-26B-A4B-it`, direct vLLM, non-thinking, temperature zero.
- Scheduler: `msc,llm`, excluding `oat12`; one A100.
- Ten deterministic posterior cells from fresh smoke seed `24127`: six uncollected
  field states and four collected-specimen states.
- K4 machine-assigned roots. Before collection, one root is always
  `collect:specimen` and the other three are the strongest distinct immediate field
  queries. After collection, the strongest four immediate query roots are assigned.
- The model chooses only integer indices for branch-contingent legal follow-ups.
- Collection follow-ups are restricted to the 17 newly unlocked specimen features.
  Field-query follow-ups cannot spend the terminal strategy step on collection.
- The prompt exposes decoded feature/outcome names and exact edible/poisonous branch
  probabilities. It exposes no EIG, policy score, hidden row, truth class, or ranking.
- One validation retry is permitted. Exact parser/scorer validation and all smoke
  metrics make zero additional LLM calls.

S0 passes only if all ten cells resolve, every root and follow-up is legal and
complete, the fixed collection root appears in every uncollected h2 cell, there are
zero reasoning tokens and forced exits, and no rollout/scoring LLM call occurs.
Invalid first responses corrected within the single frozen retry are retained and
reported but do not fail S0. No proposal-quality statistic is a gate at S0.

A pass authorizes only a separately preregistered fresh-seed proposal-quality gate.
A format failure permits implementation repair followed by a new smoke registration;
it does not authorize inspecting or reusing partial policy endpoints.
