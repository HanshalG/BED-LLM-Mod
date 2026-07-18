# 20 Questions Coverage-Dynamics Probe

**Status:** preregistered before any calls from this probe. This is a bounded
mechanistic screen, not a policy or endpoint comparison.

## Question

Does the existing 20 Questions belief-generation and filtering pipeline create
candidate-dependent **future truth coverage** that is not well represented by
the current one-step EIG score?

## Fixed Procedure

- Model: non-thinking `deepseek/deepseek-v4-flash` through OpenRouter for both
  questioner and answerer, temperature `0`.
- Ten target-conditioned, one-observation histories drawn in fixed config order
  from the ten listed animal targets; seed `1304`. Each state starts with a
  normal environment initial belief generation, takes one normal candidate
  question, obtains the answer from the normal answerer path, and takes the
  normal belief update.
- At each resulting state, retain the first three unique normal candidate
  questions. Compute their immediate EIG and predictive `p(Yes), p(No)` with
  the production likelihood scorer.
- For every candidate and each binary branch, call the production
  `_future_beliefs_for_answer` path. It runs normal generated-hypothesis
  regeneration, animal-name validation, and history-consistency filtering.
- The target animal is compared with the returned branch support only. It is
  never added to a belief support, prompt, likelihood score, candidate score,
  or belief-generation call. The answerer necessarily receives the target to
  produce the one real bootstrap observation; that is the environment's normal
  observation interface.
- Primary descriptive values: per-state range of expected truth coverage,
  expected-coverage regret of the immediate-EIG maximizer, and Spearman
  correlation between candidate immediate EIG and expected truth coverage.

## Budget and Failure Rules

- Config: `configs/config_animals_coverage_dynamics_openrouter.yaml`.
- Command: `set -a; source .env; set +a; PYTHONPATH=. python
  scripts/animals_coverage_dynamics.py --output-dir
  results/nonmyopic/animals_coverage_dynamics/20260718`.
- The run-level OpenRouter accounting cap is `$0.50`, with `$0.35` projected.
  The shared ledger cap remains `$40`.
- The run fails closed and reports no conclusion if it cannot collect ten usable
  states within thirteen attempts, if a request fails after bounded retries, or
  if the budget tracker rejects further spend.

## Read

This screen has no promotion claim. It just decides whether candidate-dependent
support survival is large enough to justify a matched-compute `d2` policy that
models epistemic dynamics. A near-zero spread is evidence against this route;
nonzero spread with substantial regret or weak rank agreement motivates the
next policy implementation and a separately preregistered paired endpoint run.
