# Bongard Luna Terminal Task-Batch Amendment

Date frozen: 2026-08-06, before any Bongard model request, candidate endpoint,
or scientific endpoint was accessed.

## Motivation

Interface `-5` gives every terminal history within a task one common requested
seed and places distinct dynamic/history-blind terminal histories adjacently.
The OpenRouter adapter correctly transmits those per-request seeds, but it
internally dispatches at most 24 requests together. A contiguous task group
beginning near an index divisible by 24 could therefore be split between two
provider dispatch waves. If a provider only approximately honors seeds, that
split weakens the intended temporal pairing.

This is a transport-level variance correction. It does not change any task,
prompt, visible image, label, hypothesis schema, policy, score, endpoint,
threshold, model request, token cap, or dollar cap.

## Frozen Terminal Dispatch

Terminal cases remain ordered contiguously by task, with distinct
`dynamic_depth2` and `history_blind_depth2` histories first and second. Before
generation, greedily pack complete task groups into dispatch batches of at
most 24 requests. A task group may not be split across batches. Since each
task has 4--10 distinct terminal histories, every task fits in one batch.

The exact ordered dispatch manifest records, for every batch:

- zero-based batch index;
- inclusive start and exclusive stop request indices;
- request count; and
- ordered opaque task IDs.

Execution invokes the model separately for each recorded terminal batch.
Development checkpoints only after a complete batch. Mechanics independent
replay consumes the same number and size of terminal batches and fails on an
extra, missing, resized, overlapping, noncontiguous, or over-24 batch.

The public terminal diagnostic and private raw manifest replay-gate:

- exact full request coverage by legal batches;
- one dispatch batch for every complete task-level terminal group;
- one dispatch batch for each distinct dynamic/history-blind terminal pair;
- all interface-`-5` common-seed and ordering invariants; and
- unchanged request counts and cost accounting.

Mechanics and development interfaces advance to `-6`. The unopened
development manifest and August 10 wrapper hash are rebound before any paid
request.
