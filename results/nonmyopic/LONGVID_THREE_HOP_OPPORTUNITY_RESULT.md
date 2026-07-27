# LongVidSearch Three-Hop Opportunity Result

## Decision

The exact preregistered three-hop construction **fails**. No OpenRouter smoke
is authorized.

The audit completed all 40 frozen opportunity tasks with zero model calls and
zero spend. The public artifact SHA-256 is
`fa3f2f69de10ea7f55d4b6dcb00df80b65044c4613e16d65d29d4a6fd3e363a3`.

## Frozen-Gate Results

| Gate | Result | Pass |
|---|---:|:---:|
| Complete tasks | 40/40 | yes |
| Diverse-root tasks | 40/40 | yes |
| Depth-three gain tasks | 36/40 | yes |
| Ordered `0→1→2` chains | 5/40 | **no** |
| Mean oracle triple coverage | 0.7083 | yes |
| Mean coverage gain | 0.4000 | yes |
| Strict opportunities | 0/40 | **no** |
| Strict total gap | 0 clips | **no** |
| Mean strict answer sacrifice | 0.0000 | **no** |

The conjunction fails and the separately proposed 10-call serving smoke is
forbidden.

## Mechanism Diagnosis

The interface retrieves useful sequential evidence but does not implement the
preregistered source-order prerequisite:

- greedy and oracle roots are identical on 34/40 tasks;
- greedy retrieves all three clips on 7/40 tasks and oracle on 11/40;
- root choice changes on only six tasks;
- all six changed roots have lower direct-answer support;
- four of the six changed roots improve from two clips to all three;
- none of the six oracle roots begins at source evidence position 0; and
- only five tasks contain any complete `0→1→2` trajectory.

The six changed-root cases are therefore descriptive evidence of a possible
semantic first-action tradeoff, but they do not satisfy the prospectively
required causal path. Their better roots generally begin at evidence position
1, showing that LongVidSearch's released evidence-list order is not the same as
the prerequisite retrieval order induced by this BM25 interface.

The result is not repaired by relabeling position 1 as the bridge, dropping the
ordered-chain requirement, selecting the six changed-root tasks, or changing
the candidate grammar after outcomes. Any broader "lower direct answer,
higher final coverage" protocol would be a new development-derived method and
would require a separately frozen confirmation on caption-unopened videos.

## Category Check

| Category | Ordered chains | Same root | Greedy full | Oracle full | Mean oracle coverage | Mean gain |
|---|---:|---:|---:|---:|---:|---:|
| Causal Inference | 3/20 | 16/20 | 4/20 | 7/20 | 0.7167 | 0.4333 |
| State Mutation | 2/20 | 18/20 | 3/20 | 4/20 | 0.7000 | 0.3667 |

The failure is not isolated to one category.

## Scope And Freshness

Caption text was loaded only for the 40 frozen opportunity videos. The 20
three-hop development videos, 40 three-hop reserve videos, and 22
caption-only fresh videos remain caption-unopened.

Calls: `0`. Cost: `$0`. OatML/Slurm: `0`.

