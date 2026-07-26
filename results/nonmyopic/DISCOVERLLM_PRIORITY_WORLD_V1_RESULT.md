# DiscoverLLM Creative Priority-World V1 Result

## Verdict

The zero-call structural gate **fails**. Creative Writing alone contains 98
eligible uninspected artifacts, below the prospectively encoded minimum of 100.
No simulator, scorer, endpoint, or OpenRouter call is authorized for this V1
line.

## Frozen Construction

- Pinned DiscoverLLM code: `a9eb2846`
- Pinned Creative Writing Parquet:
  `e8f76a47447b442e59e0d76718228b94bba1a11d8f7f653cdf1498bb0adf7843`
- Selection seed: `24411`
- Excluded after source inspection:
  `artifact_1`, `artifact_11`, `artifact_151`, `artifact_159`
- A candidate artifact must expose exactly two distinct released turn-one
  completions and at least four currently hidden hierarchy roots, each with at
  least three nodes and depth at least two.
- Four roots are selected without reading their text and treated as a uniform
  latent priority-world prior.
- Released scores and winner labels are neither loaded nor emitted.

## Structural Result

| Quantity | Observed |
| --- | ---: |
| Released artifacts with turn-one candidate pairs | 360 |
| Eligible artifacts | **98** |
| Required | **at least 100** |
| Eligible artifacts with an interactive candidate | 98 |
| Candidate actions per artifact | 2 |
| Priority worlds per artifact | 4 |

All secondary mechanics checks pass. The three frozen mechanics IDs are
`artifact_257`, `artifact_38`, and `artifact_448`; their ordered split hash is
`e9ac1e814b19181db8707091053a3c1b318fd13ff7b40cd0728e825a3b64eb7d`.
Their world-selection hash is
`aaf94f7c9f48dd962c43080053c95f340f10327f5364905f434c7e77bf3d75ab`.

The two-artifact shortfall is not scientifically meaningful, but moving the
threshold after observing it would invalidate the gate. V1 therefore remains
failed. A separately frozen V2 may test the broader released source across all
three domains and each artifact's earliest available candidate turn.

OpenRouter calls/cost: `0 / $0`. OatML jobs: `0`.
