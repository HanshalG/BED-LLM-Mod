# tau-Knowledge GPT-5.4 Rank-Ensemble Confirmation

## Status

Frozen before any confirmation-ensemble response. The method was developed
after inspecting the first three same-task scorer retests. This is a fresh
model-execution block on the same open 20 tasks, not fresh-task generalization
or a pristine held-out method test.

## Development Disclosure

The frozen development inputs are the three GPT-5.4 retest artifacts:

- `932ca6c046a0c8bd6cd9c48786600e549da3f3dd10e4eb80ddf540b52b8f4135`
- `9c81e75209c095f508f3838ac4da72bf6b5693aa2924f0d005c00e4009f72877`
- `4ef14acb5121629510d887c18414d9881f19cc7e15503365ac105013026fc4ec`

Raw mean and median aggregation were inspected first and each missed one
original endpoint gate. Range-normalized, rank-averaged, and z-normalized
scores were then inspected; all three passed the open development endpoint.
Rank averaging is selected because it removes arbitrary cross-call score scale
without using endpoint values in the aggregation rule.

The development rank ensemble has root accuracy `.6736`, root gain over
myopic `+.0909`, focused accuracy `.7610`, and endpoint `30` versus myopic
`26`, with `5/2/13` wins/losses/ties. These values are development evidence
only.

## Frozen Ensemble

For every myopic root, non-myopic root, and focused continuation prompt:

1. Run GPT-5.4 three physically separate times using the unchanged explicit
   non-reasoning V3.1 scorer.
2. Within each response score vector, replace every candidate score by its
   midrank: the number of lower scores plus half the number of other tied
   scores.
3. Average each candidate's midrank across the three responses.
4. Select the largest average rank, breaking ties by original candidate order.

The joint non-myopic continuation control aggregates each response's reported
best-followup index by majority vote, again breaking ties by original order.
No rationale text enters aggregation.

## Confirmation Block

Run three new, sequential 140-call scorer replicates on the exact frozen V3.1
confirmation trees:

- 20 myopic root prompts per replicate;
- 20 full-tree non-myopic root prompts per replicate; and
- 100 focused continuation prompts per replicate.

All prompts, parsers, trees, beliefs, queries, documents, endpoints, and
temperature remain unchanged. No tree or retrieval regeneration occurs.

## Frozen Gates

All gates must pass:

- all three confirmation replicates complete exact 140 calls with zero
  reasoning, malformed responses, forced exits, or repairs;
- total physical requests equal `420` and adapter-attributed cost is at most
  `$6.75`;
- the single confirmation ensemble passes every original V3.1 confirmation
  gate, including:
  - root accuracy at least `.60` and gain over compute-matched myopic at least
    `.05`;
  - focused accuracy at least `.60`, optimal rate at least `.70`, mean regret
    at most `.30`, and selected-root loss at most 5;
  - at least four wins, at most two losses, and total gain at least four over
    the rank-ensembled myopic policy;
  - at least six wins, at most four losses, and total gain at least five over
    frozen random;
  - at least four documents gained over the majority-aggregated joint
    continuation control; and
  - root accuracy above `.5455`, focused accuracy above `.6316`, and endpoint
    at least `25`, beating the strongest frozen nonsemantic controls;
- confirmation and development rank ensembles select the same non-myopic root
  on at least `.60` of tasks; and
- they select the same focused continuation on at least `.75` of all roots.

No individual replicate must pass the endpoint gate; the ensemble is the
preregistered policy and the three calls are its matched compute.

## Interpretation

Passing supports a reproducible same-model, same-task semantic policy after a
fixed scale-invariant aggregation of LLM scores. It does not add task
generalization, establish cross-model transfer, or repair the
refreshed-belief-alignment null. Failure closes score ensembling on this
cohort without changing ensemble size, normalization, thresholds, prompts, or
tasks.

## Budget

Projected cost is about `$2.00`; hard cap is `$6.75`. The launch proceeds only
if the live balance remains at least `$31.75`, preserving the protected `$25`
reserve even at the cap. OatML remains paused.
