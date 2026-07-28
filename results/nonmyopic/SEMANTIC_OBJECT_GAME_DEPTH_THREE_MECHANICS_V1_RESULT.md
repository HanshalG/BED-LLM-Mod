# Semantic Object Game Depth-Three Mechanics V1 Result

Date: 2026-07-29

## Decision

V1 failed at the structured-output transport boundary before any model response,
accepted request, token, or charge. No semantic support, tree, policy score, or
endpoint exists.

Azure rejected the JSON Schema because the `members` array used the unsupported
keyword `uniqueItems`:

```text
Invalid schema for response_format 'semantic_object_concepts':
'uniqueItems' is not permitted.
```

The authenticated credits endpoint was unchanged at `$180.00` total and
`$161.579636306` used. There is no raw-response artifact.

## Admissible V2

The original preregistration permits a transport-distinct successor after a
diagnosed serving failure. V2 may remove only `uniqueItems` from the provider
schema. The existing exact parser continues to reject duplicate member IDs, so
accepted semantics are unchanged.

Models, prompts, universe, seeds, temperature, call count, support filtering,
planning, endpoint, gates, and budget remain frozen. V2 must be committed before
its first response.

Public failure artifact:
`results/nonmyopic/semantic_object_game_depth_three_mechanics/semantic-object-game-depth-three-mechanics-20260729T010000Z/FAILURE.json`.

OpenRouter cost: `$0`. OatML, Slurm, SSH, and cluster use: `0`.
