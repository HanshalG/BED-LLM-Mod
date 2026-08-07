# RegretBench Matched-Utility Myopic Amendment

Date: 2026-08-07

Status: frozen before any RegretBench policy response or endpoint.

## Problem

The dynamic depth-two selector minimizes expected terminal truth-group Brier,
while the original `myopic_width` selector maximizes immediate reply entropy.
Both are proper information criteria, but with more than two hypothesis groups
they need not rank the same four questions. Therefore dynamic-versus-
`myopic_width` can combine a horizon effect with an entropy-versus-Brier
objective change.

## Matched-Utility Control

Add `myopic_brier` as a primary policy. For every candidate first question and
each initial generated hypothesis treated as truth:

1. use that hypothesis's aligned predicted first reply;
2. condition the initial eight-particle support on the exact normalized reply;
3. compute posterior mass on hypotheses whose final answer matches the treated
   truth under the frozen lexical matcher; and
4. score `(1 - posterior truth-group mass)^2` and frozen-floor log loss.

Average over treated truths using their normalized initial prior weights.
`myopic_brier` selects the minimum expected one-step Brier, breaking ties by
lowest original question index. It then uses the exact same realized
answer-conditioned refresh, maximum-EIG second question, official environment
reply, aligned terminal endpoint, validity penalties, and task-level common
random numbers as every other primary policy.

This selector uses only the shared initial support. The already generated full
tree remains shared across all policies, so the experiment-level compute and
call schedule are unchanged. At most four distinct roots still exist per task;
adding a sixth policy cannot increase realized-path requests.

## Mandatory Mechanics

`myopic_brier` is included in every existing per-primary-policy support,
action-novelty, first-reply alignment, second-reply match, privacy, endpoint,
and independent-replay check. Public results additionally report all four
one-step Brier root risks and the selected root. No hidden truth enters
selection.

## Mandatory Science Gates

All original gates remain conjunctive. Add all of the following:

1. dynamic and `myopic_brier` roots differ on at least `16/64` tasks;
2. mean conditioned predicted terminal-Brier advantage of dynamic over the
   `myopic_brier`-selected root is at least `0.01`;
3. dynamic-minus-`myopic_brier` aligned realized Brier is at most `-0.02`;
4. its 20,000-sample paired bootstrap probability of improvement is at least
   `0.90`;
5. Brier wins exceed losses;
6. dynamic mean aligned log loss is no worse than `myopic_brier`; and
7. on changed dynamic/`myopic_brier` roots, predicted terminal-Brier advantage
   versus realized Brier advantage has Spearman at least `0.15` and bootstrap
   probability of positive correlation at least `0.80`.

The existing entropy-myopic comparison remains mandatory and is relabelled as
the `myopic_eig_width` control in prose while retaining the serialized policy
key `myopic_width`. The matched-Brier policy is the headline myopic comparator.
Failure of any new gate makes the exact development or confirmation result a
preregistered null.

## Reporting

The frozen report and paper table include `myopic_brier` alongside all prior
controls. The primary interpretation and alignment-complete diagnostic use
dynamic versus `myopic_brier`; entropy-myopic remains visible and required.
No optional, secondary, pooled, or subset result can rescue the matched-utility
conjunction.

## Scope

This prospective amendment adds no model call, prompt, generated support,
candidate, realized path beyond the existing four-root maximum, seed, task,
budget, or retry. It changes policy selection and scientific requirements only
by adding the matched-loss control before responses. Development and the
untouched confirmation cohort use the same rule and thresholds.
