# tau-Knowledge Evidence-Only Receding Continuation V2

## Motivation

V1 passed serving and ranked 291 endpoint-distinct continuation pairs at 0.7079
accuracy, but failed its old-tree development gate. The endpoint-aware audit
identified a concrete interface error: rationales sometimes credited the policy
suggested by a candidate query even when that policy was absent from the
candidate's returned documents.

V2 changes only that feature. Candidate followup query strings are removed, and
the scorer must evaluate the already returned document evidence. This protocol
is frozen before any V2 response is generated.

## Frozen interface

- Source, task splits, generated trees, model, temperature, parser, score schema,
  tie breaking, root scorers, controls, and endpoint calculation are identical
  to V1.
- GPT-5.4 runs with provider reasoning disabled.
- Each call still sees one realized root, the customer opening, initial and
  refreshed information-need hypotheses, acquired first documents, and four
  groups of candidate returned documents.
- It does **not** see candidate followup query text.
- A candidate receives credit only for concrete facts explicitly supported by
  its returned document titles and excerpts. Query intent, imagined documents,
  raw document count, and duplicate evidence receive no credit.
- Required-document IDs, full user scripts, endpoints, and evaluation actions
  remain hidden.

## Stages and gates

V2 reuses the exact V1 stage sizes and frozen thresholds.

Public serving smoke uses `task_018` and `task_008`, all five roots, for exactly
10 calls. It requires complete canonical responses, zero reasoning, score
variation on at least 8/10 roots, pairwise accuracy at least 0.55, and at least
7/10 oracle-optimal selections.

Old-tree development uses the already-unsealed 20-task V1 confirmation trees for
exactly 100 calls. Every condition must pass:

- at least 200 endpoint-distinct followup pairs;
- pairwise accuracy at least 0.60;
- at least 72/100 oracle-optimal followups;
- all-root regret at most 28 documents;
- selected non-myopic-root continuation loss at most 5 documents;
- non-myopic receding versus myopic: at least 3 wins, at most 2 losses, and
  total gain at least 3 documents;
- focused non-myopic gain over the original joint selector at least 5 documents.

Only a passing development result releases the same untouched 20-task
confirmation split selected with seed `24337`. Confirmation remains exactly 280
calls and retains all V1 gates. No response, task, query, score, threshold, or
malformed output may be repaired, replaced, or changed.

## Budget

The live balance before V2 is `$56.459063`, leaving `$31.459063` above the
protected `$25` Monday reserve. Smoke, development, and conditional confirmation
remain projected at `$0.10`, `$1.00`, and `$3.50`, with hard per-stage caps of
`$0.50`, `$3`, and `$8`. OatML remains paused.
