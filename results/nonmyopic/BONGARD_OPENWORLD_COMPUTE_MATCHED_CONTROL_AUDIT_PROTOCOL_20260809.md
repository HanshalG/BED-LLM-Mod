# Bongard OpenWorld Compute-Matched Control Audit Protocol

Frozen: 2026-08-09 (Europe/London), before any Bongard mechanics,
development, confirmation, or scientific endpoint response.

Status: **prospective zero-call audit**. The separate compute-matched myopic
amendment makes its new control part of the main frozen conjunction; this audit
remains a read-only report and cannot change gates after execution.

## Purpose

The primary `dynamic_depth2` versus `myopic_width` comparison tests non-myopic
lookahead against an online-regeneration greedy policy, but the dynamic policy
uses counterfactual branch generations when selecting its first query. This
audit makes the compute-matched evidence explicit rather than treating the
myopic comparison as compute matched.

## Frozen Controls

For every task report four paired comparisons:

1. **Compute-matched myopic ensemble:**
   `compute_matched_myopic_ensemble` averages one-step endpoint PIG across the
   ordinary root and all 16 distinct-seed, byte-identical root-prompt beliefs.
   It therefore spends the same 17 belief calls as dynamic first-query
   planning without conditioning the extra calls on possible answers.
2. **Shuffled continuation:** `shuffled_dynamic_depth2` uses the exact same
   root EIG vector and complete dynamic continuation-value multiset as
   `dynamic_depth2`, with the continuation values deterministically permuted
   across first actions. This is the strict branch-bank-compute-matched control.
3. **History blind:** `history_blind_depth2` uses one same-seed, same-batch,
   adjacent VLM generation for every conditioned dynamic branch, but hides the
   simulated answer. This is the matched branch-request-count control.
4. **Online greedy:** `myopic_width` selects each real query by one-step
   endpoint-predictive EIG and uses the same answer-conditioned regenerated
   belief after the first real answer. This is the practical greedy baseline,
   not an algorithmically compute-matched baseline.

The audit must verify from every stored tree that:

- dynamic and shuffled score maps have the same candidate support;
- each dynamic score equals root EIG plus its own stored continuation value;
- each shuffled score equals root EIG plus its stored shuffled continuation;
- shuffled continuation values are an exact multiset permutation of dynamic
  continuation values;
- each stored first query is the deterministic argmax of its score map;
- all four controls and dynamic have finite endpoint Brier and log loss.

Any mismatch fails closed before metrics are emitted.

## Frozen Metrics

On all tasks, with no selected subsets, report dynamic minus each control for:

- endpoint mean Brier;
- endpoint mean log loss.

For each vector use the existing paired-summary implementation with 20,000
bootstrap draws and fixed seeds derived from `2026280901`. Report `n`, mean,
sample standard deviation, standard error, 95% bootstrap interval, bootstrap
probability that the difference is below zero, and wins/ties/losses. Also
report first-query and final-history change counts.

Negative differences favor dynamic. The report must identify the myopic
ensemble as the strict call-matched myopic control, shuffled as the strict
continuation-bank control, history-blind as request-count matched, and myopic
as the online-greedy cost/performance baseline.

## Authorization And Interpretation

The stage-specific independent authorizer must verify mechanics, development,
or confirmation before the result is loaded for analysis. The output binds the
stage result and every block result by SHA-256 and is written once.

This report cannot alter, rescue, or veto any frozen gate, claim tier,
development authorization, confirmation authorization, classical comparison,
or paper headline. The main executors independently enforce the preregistered
myopic-ensemble gates. A favorable ordinary-myopic result without a favorable
myopic-ensemble result is not evidence of compute-matched superiority.
Conversely, a favorable matched result does not prove that no cheaper classical
method can match the policy.

The audit makes zero model calls, costs `$0`, and authorizes no paid call or
rerun.
