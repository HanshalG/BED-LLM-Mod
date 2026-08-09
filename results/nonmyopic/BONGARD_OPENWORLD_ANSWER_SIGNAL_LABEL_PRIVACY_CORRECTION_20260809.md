# Bongard Answer-Signal Label-Privacy Correction

Frozen: 2026-08-09 Europe/London, before any August 10 Bongard response,
candidate label, endpoint label, or terminal artifact was opened.

Status: **prospective zero-call implementation correction; changes no paid
request, metric, threshold, task, or authorization rule**.

## Defect

The answer-signal amendment requires a zero-call replay that opens no new
label. Its first implementation computed only from first-stage belief
responses, but its default task resolver called the general mechanics loader.
That loader materialized candidate and endpoint labels in `VisualTask` even
though the audit never read them. The numerical audit was label-independent,
but the implementation did not enforce the registered privacy boundary.

## Correction

The audit now owns a hash-bound public manifest containing only the four
already-frozen opaque task IDs, fourteen opaque image IDs, four initially
observed labels, eight candidate IDs, and two endpoint IDs per task. It creates
tasks with empty image payloads, empty `actual_labels`, and no hidden source
values. The general source/task loader is not called.

Before parsing any mechanics response, the audit rejects a task collection
unless the initial, candidate, and endpoint IDs form the exact disjoint
four/eight/two partition and both `actual_labels` and `hidden_values` are
empty. A regression test replaces the general mechanics loader with a function
that raises and still requires an exact report replay.

## Scientific Effect

None. Candidate and endpoint outcomes are not inputs to the answer-signal
metric. The same hash-bound first-stage responses, candidate pairs, predictive
MAEs, regeneration-noise comparison, thresholds, and every-task gate remain in
force. A null or malformed answer-signal result still blocks all endpoint
postprocessing and Development64. This correction only makes the existing
"opens no new label" claim mechanically true.
