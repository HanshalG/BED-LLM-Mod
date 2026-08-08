# Bongard Matched Realized-Updater Amendment

Frozen: 2026-08-08, before any Bongard mechanics, development, or
confirmation response and before any scientific endpoint was opened.

This amendment adds a matched policy that directly tests whether the LLM's
answer-conditioned intermediate belief changes the next experimental action in
a useful way. It does not change tasks, images, root or branch prompts, root or
branch seeds, endpoint definitions, existing policies, or existing gates.

## Matched Policy

The new policy is `history_blind_update_matched_first`.

For each task:

1. Use exactly the first query selected by `dynamic_depth2` and observe the
   same realized label.
2. Retrieve the already paired same-seed history-blind branch for that query
   and label. This support was generated without the first-query answer in its
   prompt.
3. Apply the same analytical Bernoulli update for the realized label to that
   blind support.
4. Select query two by endpoint predictive information gain on the updated
   blind support.
5. Regenerate the terminal belief from the resulting complete realized history
   using the same task-level seed and dispatch batch as the dynamic terminal
   history.

Thus dynamic and control share the initial support, first query, first answer,
terminal updater, endpoint, and terminal common random number. They differ only
in whether the intermediate belief support that chooses query two was generated
with the realized first answer in context. This identifies the downstream
action value of the LLM's answer-conditioned intermediate belief dynamics.

The existing `history_blind_depth2` arm remains a separate first-query planning
simulation control under a common realized updater. Its estimand and gates do
not change.

## Prospective Development Gates

The complete path-dependent development family additionally requires:

- at least 24 of 64 dynamic final histories differ from the matched updater;
- at least 24 changed second actions clear the numerical tie margin under both
  second-step score maps;
- at least one changed final history in every execution block;
- at least 3% relative mean endpoint-Brier improvement;
- paired bootstrap probability of Brier improvement at least 0.80; and
- mean endpoint log loss no worse than the matched updater.

Failure of any new gate prevents the full path-dependent tier and confirmation
authorization. Other prospective partial tiers remain descriptive only.

## Prospective Confirmation Gates

The complete confirmation conjunction additionally requires:

- at least 36 of 96 changed final histories;
- at least 36 robust changed second actions;
- at least one changed final history in every execution block;
- at least 3% relative mean endpoint-Brier improvement;
- the paired-tree bootstrap 95% interval for dynamic minus control Brier has an
  upper endpoint below zero; and
- mean endpoint log loss is no worse than the matched updater.

## Request And Budget Effect

The root and branch stages are unchanged. A matched updater path may add at
most one distinct terminal history per task:

- mechanics: at most 176 accepted responses, 180 HTTP attempts, and `$0.720`
  precharged exposure under the unchanged `$1.75` run cap;
- each 16-task development block: at most 704 accepted responses, 719 HTTP
  attempts, and `$2.876` precharged exposure;
- development plus its same-day 16-call naive block: at most `$3.004`
  precharged exposure under the account-wide `$5.00` daily cap;
- each 24-task confirmation block: at most 1,056 accepted responses, 1,078 HTTP
  attempts, and `$4.312` precharged exposure under the unchanged `$4.75` run
  cap and `$5.00` daily cap.

Only distinct final histories are dispatched. Existing task-level terminal
common-random-number and task-preserving batch gates remain mandatory.

## Allowed Claim

Only a passing complete development family may support the prospective claim
that, after the same first query and answer, answer-conditioned LLM belief
regeneration selected better second queries than same-seed history-blind belief
generation under a common terminal updater. Only a separately passing frozen
confirmation may upgrade that statement to confirmed evidence.
