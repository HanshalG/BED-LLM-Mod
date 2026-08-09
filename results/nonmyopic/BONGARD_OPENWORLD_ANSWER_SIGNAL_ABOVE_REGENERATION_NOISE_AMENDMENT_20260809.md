# Bongard Answer Signal Above Regeneration Noise Amendment

Frozen: 2026-08-09 Europe/London, before any August 10 Bongard response,
candidate label, endpoint label, or terminal artifact was opened.

Status: **prospective zero-call validity gate; changes no paid request**.

## Motivation

The existing mechanics tree pairs every answer-conditioned branch request with
a same-seed history-blind request whose prompt differs only by the simulated
answer. Existing gates verify prompt accounting, branch-label obedience, and
changes on still-unobserved images. They do not directly require the numerical
predictive-matrix shift caused by opposite answers to exceed the shift observed
between repeated same-history regenerations.

That distinction is necessary for the registered first-link interpretation. A
planner cannot be credited for exploiting answer-conditioned future beliefs if
the relevant predictive changes are no larger than support-regeneration noise.

## Frozen Audit

For each of the 32 task/candidate pairs in mechanics:

1. compute mean absolute predictive-probability change over still-unobserved
   images between the negative and positive answer-conditioned beliefs;
2. compute the same change between the paired negative-seed and positive-seed
   history-blind beliefs, which receive the identical initial-history prompt;
3. subtract history-blind change from answer-conditioned change.

The audit passes only when all of the following hold:

- all four tasks and all 32 candidate pairs replay exactly;
- every metric is finite;
- pooled answer-conditioned unobserved-prediction MAE is at least `0.05`;
- pooled answer-conditioned-minus-history-blind MAE is at least `0.01`;
- the mean advantage is strictly positive in each of the four tasks.

The thresholds are frozen before outputs. Candidate rows are descriptive and
are not treated as independent inferential replicates; the every-task condition
prevents a single mechanics task from carrying the validity gate.

## Authorization

The audit makes zero model calls and opens no new label. It runs only after an
existing mechanics pass and replays the hash-bound raw responses. A null or
malformed audit blocks Development64 even if the older paid wrapper reports its
pre-existing authorization. It cannot authorize paid work, a rerun, a rescue,
or a scientific claim by itself.

No model, prompt, response schema, seed, task, image, request count, action,
endpoint, likelihood, budget, or paid execution date changes.
