# ScholarGym Retrieval-Path Opportunity Result

## Decision

The frozen gate failed. This exact ScholarGym two-search construction is closed
before any LLM use. No serving smoke, policy-development run, or holdout
evaluation is authorized.

## Frozen Inputs

- ScholarGym repository commit:
  `cb1c5fc7c796308bef353b18549a92e2739f5b96`
- Hugging Face revision:
  `be7d917ddac3cd6f2878f81160965f914dab3706`
- query benchmark SHA-256:
  `869f507eacb7f554b8f4e6dc65ea97b01f95140ba5aa86a5119b330c41b9d551`
- 570,206-paper corpus SHA-256:
  `e1e570321e22bf59784d46a9dd28239350d7ad1ddda213eee2f5730710f239e8`
- opportunity/development/holdout ID hashes:
  `6f930b3cd8a98fa565371150a87ddbb07143591a14f437379bf1d0aa9b5dbb53`,
  `dcf14914b3516afe2362915422b99c44beeae16ea4596778dad31f682ac6efdf`,
  and
  `04fd9f531470f94f42d89f56cbc1f676ffa28466e6768ca617a140bbc896f15e`
- audit artifact SHA-256:
  `a8857462645f2b6f35ef616656fc14a33b35b8448d96248f1ba8f8d7906c210c`

The audit opened only the 40 opportunity queries. The 24 AutoScholar test
development queries and 24 human-written RealScholar holdout queries remain
sealed.

## Results

| Metric | Frozen requirement | Result | Pass |
|---|---:|---:|:---:|
| Complete tasks | 40 | 40 | yes |
| Tasks with at least 3 distinct root top-1 papers | >=30 | 40 | yes |
| Tasks with a depth-two recall gain | >=15 | 9 | no |
| Mean oracle pair recall | >=.15 | .2184 | yes |
| Mean pair-recall gain | >=.03 | .0849 | yes |
| Strict non-myopic opportunities | >=5 | 0 | no |
| Total strict ground-truth-paper gap | >=5 | 0 | no |
| Mean normalized strict gap | >=.15 | 0 | no |

Mean best-immediate recall was `.1336`; mean oracle two-search recall was
`.2184`. Depth two therefore has useful retrieval capacity, but not the
required action-order tradeoff.

The decisive diagnostic is exact:

- greedy and two-step oracle roots are identical on `40/40` tasks;
- 26 tasks retrieve zero ground-truth papers at the best immediate root;
- 13 retrieve one and one retrieves two;
- 31 tasks gain no paper at depth two, seven gain one, and two gain two; and
- every depth-two gain belongs to the same root selected greedily.

Thus observation-conditioned continuation helps expand retrieval after a good
first query, but no task requires a lower-immediate enabling first query to
reach a better final paper set.

## Interpretation

ScholarGym validates iterative semantic search, not non-myopic BED under this
construction. Its published multi-iteration gains are compatible with repeated
myopic improvement. The LLM can formulate useful followups, but a full-tree
planner has no root-selection advantage to learn when the oracle root always
coincides with the greedy root.

This reproduces the structural lesson from DR3-Eval on a much larger,
source-grounded corpus:

> path dependence in the second action is insufficient; the environment must
> force an enabling first action that is worse immediately and better after
> adaptation.

Changing top-k, adding action costs, weighting papers, or constructing a new
hard subset after seeing these outcomes would define a new endpoint and is not
a repair of this gate.

## Cost And Next Constraint

- OpenRouter requests: `0`
- OpenRouter cost: `$0`
- OatML/cluster jobs: `0`
- authenticated balance before the audit: `$33.574043`
- reserve through Monday: at least `$25`
- maximum new pre-Monday spend: `$8.50`, still entirely unused

The next candidate must expose prerequisite or enabling structure in its source
labels or transition rules before any model call. Generic iterative retrieval
benchmarks should not receive another paid smoke without that exact
lower-immediate, higher-final gate.
