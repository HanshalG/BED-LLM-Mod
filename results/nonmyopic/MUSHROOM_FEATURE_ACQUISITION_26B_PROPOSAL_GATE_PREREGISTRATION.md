# Mushroom Feature Acquisition 26B Proposal Gate Preregistration

Registered 2026-07-23 after S0 v4 passed and before any Mushroom proposal-quality
score was computed or inspected.

## Purpose

Test whether non-thinking Gemma 4 26B proposes useful branch-contingent continuations
on the qualified semantic feature-acquisition task. This is a same-state proposal
gate, not a policy endpoint. A pass is required before spending on a fresh paired
policy confirmation.

## Frozen Design

- Model/interface: the S0-v4 contract, `google/gemma-4-26B-A4B-it`, direct vLLM,
  non-thinking, temperature zero, K4 machine-fixed roots, root-keyed integer arrays,
  one validation retry, and a 128-token completion cap.
- Fresh seed `24131`; 32 distinct realistic posterior cells: 16 uncollected and 16
  collected. Hidden catalog rows are sampled without replacement.
- Uncollected cells follow zero to two queries from a seeded, catalog-independent
  diverse legal-field schedule and are retained only when exhaustive endpoint-aligned
  d2 strictly prefers collection over the d1 root.
- Collected cells follow collection plus zero to three queries from a seeded,
  catalog-independent diverse legal-feature schedule and retain positive target
  entropy. This avoids reducing the collected set to a handful of odor-collapsed
  histories while never selecting on LLM output. Both phase schedules are frozen by
  seed before responses. Cell construction and filtering
  make zero LLM calls.
- Each accepted model response supplies one h2 policy for each of four fixed roots.
  Exact expected cumulative post-action class entropy scores all four; the verifier
  selects the lowest-cost proposal.

## Frozen Controls

- `matched_random`: one uniformly random legal continuation per positive-probability
  branch under the identical four roots, with the same four-candidate exact verifier.
- `shared_d1_exact_continuation`: choose the lowest immediate-entropy root among the
  same roots, then give that root its exact optimal branch continuations. This is a
  strong myopic-root control rather than a weak arbitrary continuation.
- `exhaustive_d2`: exact best root and exact branch continuation over every legal
  action, used as a ceiling and recovery denominator.

All controls, likelihoods, posteriors, scores, and bootstrap samples are exact and
make zero LLM calls. The hidden row and all scores/rankings remain absent from prompts.

## Frozen Endpoints And Gate

Use 5,000 paired bootstrap replicates. S1 passes only if every mechanics check passes
and all four proposal conditions hold:

1. Overall `matched_random - LLM` h2 cost has a strictly positive 95% lower bound.
2. On the 16 uncollected opportunities, `shared_d1_exact_continuation - LLM` h2 cost
   has a strictly positive 95% lower bound.
3. The exact verifier selects the collection-root LLM policy in at least 75% of those
   uncollected cells.
4. Mean recovery `(shared_d1_cost - LLM_cost) / (shared_d1_cost - exhaustive_d2_cost)`
   is at least 0.60 on the uncollected cells.

Mechanics require 32/32 resolved cells, balanced/distinct construction, legal complete
policies, paired identical roots for LLM/random, zero reasoning tokens, zero forced
exits, and zero scoring/rollout LLM calls. Invalid first responses corrected within
the frozen retry are reported but do not fail the gate.

A pass authorizes only a separately preregistered fresh-seed 30-trial, eight-round
paired policy confirmation with entropy AUC primary, truth-log AUC corroboration, and
shared-d1, exhaustive-d1, matched-random, and exhaustive-d2 controls. A failure stops
the Mushroom LLM policy line without threshold tuning.
