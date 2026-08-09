# Bongard OpenWorld Random-Strategy Control Protocol

Frozen: 2026-08-09 13:22 Europe/London, before any August 10 Bongard model
response, candidate label, endpoint label, or terminal artifact was opened.

Status: **prospective zero-call descriptive audit; authorizes no model request**.

## Motivation

The paid Bongard protocol already freezes and executes a random two-query policy
for every task, and every combined result retains its task-level endpoint scores.
The methods paragraph names that control, but the paper-facing summaries do not
currently report a paired dynamic-versus-random interval. This protocol prevents
that baseline from being selectively omitted after outcomes are known.

It changes no model, prompt, response schema, task, split, seed, action, policy,
likelihood, endpoint, terminal updater, request count, budget, gate, claim tier,
confirmation authorization, or headline rule.

## Frozen Random Policy

For each task, the existing policy draws two distinct candidate IDs without
replacement using Python `random.Random` with seed

```text
2026081022 + int.from_bytes(sha256(task_id)[:8], "big")
```

and the task's canonical sorted candidate IDs. It does not use an LLM score,
candidate label, endpoint label, or realized first answer to choose either query.
Its terminal belief request uses the same task-level common random number and
history-key deduplication as every other policy. The audit must independently
replay both random choices from each stored task ID and root score support.

## Estimand

For every authorized task in the stage, compute

```text
endpoint_metric(dynamic_depth2) - endpoint_metric(random)
```

for mean Brier score and mean log loss. Negative values favor dynamic depth two.
Use all tasks: four for mechanics, 64 for development, and 96 for confirmation.
Do not select a changed-action or favorable subset.

For each metric report `n`, mean difference, sample standard deviation, standard
error, a 20,000-draw paired task-bootstrap 95% interval, bootstrap probability
that the mean is below zero, and wins/ties/losses. Use bootstrap base seed
`2026080902`, adding the metric index. Also report first-query and final-history
change counts.

The audit must fail closed unless the existing stage authorizer independently
replays the exact mechanics/development/confirmation artifact before the stage
result is loaded. It must verify unique task IDs, finite endpoint values, complete
dynamic and random policies, two distinct valid random choices, exact replay of
the frozen random choices, and finite paired summaries.

## Interpretation

This is a sanity baseline, not a compute-matched causal contrast. A dynamic win
over random does not establish non-myopia; the myopic, history-blind, shuffled,
matched-updater, and classical controls remain necessary. A loss or null versus
random cannot be hidden, but this audit cannot change, rescue, or veto a frozen
gate or claim tier. Development and confirmation remain separate and never pool.

The output makes zero model calls, costs `$0`, and authorizes no paid call, rerun,
repair, endpoint opening, development stage, confirmation stage, or paper claim.
