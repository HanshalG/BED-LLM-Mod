# BrowseComp-Plus Terminal Semantic Development Gate

## Result

The frozen ten-task untouched development gate fails closed at the intermediate
belief-serving layer, before terminal beliefs or scientific endpoints.

GPT-5.4 nonreasoning completed all 10 initial calls and all 60 first-observation
belief-refresh calls. The initial responses all parse. Fifty-nine of 60 refresh
responses parse; task 562, root 0 assigns zero mass to its eighth candidate:

```text
H08|0|Susan Magarey
```

The frozen intermediate grammar required eight strictly positive integer
weights summing to 100, so this response is invalid. The run stopped after
exactly 70 requests and cost `$0.3509975`. It used zero retries, reasoning
tokens, forced exits, repairs, or reissues. No terminal calls were made and no
truth, evidence, or gold endpoint was computed.

## Interpretation

This is a serving failure, not an adverse policy result. The mechanics smoke
prospectively accepted nonnegative terminal masses because an output
distribution may assign zero probability; the untouched runner inherited the
older strictly-positive grammar for its intermediate beliefs. That
inconsistency is now visible.

The preregistration explicitly prohibited parser relaxation or a rerun after
failure, so this development block is closed. The strong five-task mechanics
result remains directional evidence only; independent efficacy is unmeasured.

Public failure SHA-256:
`be4d6411405a7960b8da4f2cf1195e5c17fae468ba4118f4d5c4331f9e187fc8`.

Private raw-response SHA-256:
`3b1af4b9ec34072a8aa256cb267149ea6cc17187e90f1509e1b173e9c3a1d5ab`.
