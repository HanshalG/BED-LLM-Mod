# tau-Knowledge GPT-5.4 Scorer Test-Retest

## Status

Frozen before any test-retest response. The source trees and endpoints are
already open from the V3.1 confirmation, so this is a same-task reproducibility
test, not a fresh holdout or an independent generalization result.

## Question

Does the complete GPT-5.4 semantic scorer reproduce its ranking and endpoint
advantage across independent OpenRouter executions, despite the decision
variation observed in the refreshed-belief alignment ablation?

## Frozen Inputs

- The exact 20-task V3.1 confirmation artifact, SHA-256
  `f8b120b0d2b98f9f10eb0ef9251262a6a41b20d4d90d724ee4926c141ec275ae`.
- The exact nonsemantic-control analysis, SHA-256
  `d1d43ecf0f09fd5874e9badd6cff4e9ae5c3da882a3ea408b816225c2d2b29f4`.
- The same customer openings, generated path-dependent beliefs, queries,
  returned documents, exact required-document endpoints, parser, prompts, and
  deterministic tie breaking as V3.1.
- GPT-5.4, temperature zero, explicit `reasoning_effort: none`, 4,096 output
  tokens, and OpenRouter concurrency up to 256.

No tree, query, retrieval result, belief state, task, or endpoint is
regenerated.

## Repeats

Run three physically separate scorer replicates. Each replicate makes exactly
140 requests:

- 20 myopic root scores;
- 20 full-tree non-myopic root scores; and
- 100 focused continuation scores.

Each repeat receives a fresh adapter and private raw checkpoint. Replicates run
sequentially. A malformed response, reasoning token, missing request, budget
failure, or runtime error fails the experiment closed; no response is repaired
or reissued for scientific reasons. Network retry behavior remains the frozen
adapter behavior.

## Frozen Gates

All gates must pass:

- all three replicates complete exact 140 physical requests with zero reasoning;
- at least two of three pass every original V3.1 confirmation and strongest
  nonsemantic-control gate;
- mean non-myopic root pairwise accuracy is at least `.60`;
- mean non-myopic-minus-myopic root accuracy is at least `.05`;
- mean focused continuation pairwise accuracy is at least `.60`;
- at least two replicates achieve both total endpoint gain at least `+4` over
  their freshly rescored myopic control and endpoint total at least `25`;
- mean pairwise agreement of selected non-myopic roots across the three
  replicate pairs is at least `.50`;
- mean pairwise agreement of selected focused continuations across all roots is
  at least `.65`; and
- total physical requests equal `420`, reasoning tokens equal zero, and total
  cost is at most `$6.75`.

Endpoint differences are reported per task and replicate. No confidence
interval treats the three repeated observations of one task as three
independent tasks.

## Interpretation

Passing supports scorer test-retest robustness for the frozen GPT-5.4 policy.
It does not expand task-domain generalization, establish cross-model transfer,
or reverse the refreshed-belief alignment null. Failure means the current
headline is sensitive to provider/model execution noise and must be narrowed.

## Budget

Projected total cost is about `$4.20`; hard cap is `$6.75`. The live balance
must exceed the protected `$25` reserve by at least `$6.75` before launch.
OatML remains paused.
