# Bongard Luna Terminal Common-Random-Number Amendment

Date frozen: 2026-08-06, before any Bongard model request, candidate endpoint,
or scientific endpoint was accessed.

## Motivation

The paired branch control already gives each answer-conditioned request and its
history-blind mate the same requested seed. The terminal belief requests did
not: every distinct two-query history received a different seed. With one
terminal belief draw per history, a policy-level Brier difference could
therefore contain avoidable model-seed luck in addition to the effect of the
selected history.

This is a variance and attribution correction. It does not change any task,
visible image, label, prompt content, hypothesis schema, policy, endpoint,
decision rule, gate threshold, number of requests, or budget.

## Frozen Terminal Pairing

Within each task, every distinct terminal history receives the same requested
model seed. Different tasks retain distinct deterministic seeds. Terminal
histories are ordered with `dynamic_depth2` first and
`history_blind_depth2` second whenever their histories differ, followed by the
other policy histories and then any remaining all-first-action histories.
Duplicate histories still share one request.

The public result and private raw manifest record and replay-gate:

- one common requested seed for every terminal history within a task;
- distinct terminal seeds across tasks;
- contiguous task-level terminal request groups;
- adjacent dynamic/history-blind terminal requests when their histories
  differ; and
- exact dynamic/history-blind seed equality.

Provider seed compliance is not assumed. The design requests common random
numbers and uses adjacency to reduce residual temporal variation; the observed
model response remains the deployed stochastic belief process.

An adversarial seed-sensitive unit test demonstrates the threat: history-level
seeds can induce an arbitrary terminal policy difference even when history has
no effect, whereas a task-level common seed makes that synthetic difference
exactly zero.

Mechanics and development interfaces advance to `-5`. The unopened development
manifest and August 10 wrapper hash are rebound before any paid request.
