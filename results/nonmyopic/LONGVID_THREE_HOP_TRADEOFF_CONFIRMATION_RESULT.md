# LongVidSearch Three-Hop Tradeoff Confirmation Result

## Decision

The prospective reserve confirmation **fails** its frozen conjunction. No
OpenRouter serving smoke is authorized.

The audit completed all 40 reserve tasks with zero model calls and zero spend.
The public confirmation SHA-256 is
`6d218c5bf681f12e7e08a9d75afa05f66b7942e7ba19eaea176d2feb03d41f4e`.

## Frozen-Gate Results

| Gate | Result | Pass |
|---|---:|:---:|
| Complete tasks | 40/40 | yes |
| Diverse-root tasks | 40/40 | yes |
| Depth-three gain tasks | 36/40 | yes |
| Mean oracle triple coverage | 0.6500 | yes |
| Mean coverage gain | 0.3833 | yes |
| Strict semantic tradeoffs | 4/40 | **no** |
| Strict total gap | 4 clips | **no** |
| Mean strict direct-answer sacrifice | 0.4574 | yes |

The frozen prevalence and total-gap thresholds were both five. Missing each by
one does not authorize a rerun, pooled pass, threshold change, or LLM stage.

## Replication Interpretation

The development diagnostic found 6/40 strict tradeoffs and six clips of total
gap. The reserve block independently finds 4/40 and four clips. The mechanism
therefore recurs, but below the prospectively required prevalence.

The four reserve cases are heterogeneous:

- three are Causal Inference and one is State Mutation;
- two improve necessary-clip count from two to three;
- two improve from one to two;
- oracle roots begin at evidence positions 1, 2, 0, and non-gold; and
- sacrifices are `0.2941`, `0.4545`, `0.4444`, and `0.6364`.

This supports a descriptive statement that lower-immediate roots can unlock
better three-search retrieval on about a tenth of these tasks. It does not
support the preregistered prevalence claim, a source-ordered mechanism, or an
LLM policy experiment.

## Category Summary

| Category | Tradeoffs | Same root | Greedy full | Oracle full | Mean oracle coverage | Mean gain |
|---|---:|---:|---:|---:|---:|---:|
| Causal Inference | 3/20 | 17/20 | 4/20 | 6/20 | 0.7167 | 0.4333 |
| State Mutation | 1/20 | 19/20 | 3/20 | 3/20 | 0.5833 | 0.3333 |

## Scope And Freshness

The 20 three-hop development videos and 22 caption-only fresh videos remain
caption-unopened. They are not used to rescue or further tune this route.

Calls: `0`. Cost: `$0`. OatML/Slurm: `0`.

