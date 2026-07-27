# LongVidSearch Bridge-Path Opportunity Result

## Decision

The frozen two-hop gate failed. No LLM serving smoke or paid policy run is
authorized for this exact construction.

The result is structurally closer to a useful non-myopic environment than DR3
or ScholarGym, but the two-hop task remains predominantly order-symmetric:
greedy roots can usually backtrack to the bridge clip on their second search.

## Frozen Inputs

- LongVidSearch repository commit:
  `4aa5620e06ca1bc5cc7cfce2b3e3ed0a5ae82d4e`
- QA SHA-256:
  `370711f1299202cd40d559440859476bf3a2d2ca74540dad5226786558ee123b`
- caption SHA-256:
  `0f2ce94265b7050eaee5de239a760c1a0f0762c1754cbb3bd7b45d00774b6c68`
- opportunity row hash:
  `092b0337cad5d0fce436e1378640bd1e8580a2fe29953a25e189593ba17bbbb6`
- audit artifact SHA-256:
  `55c4aec1c9d6a70157fda37437bf40547e07669504d5215a8b1e2c532ec28d11`

The caption reader returned only the 40 opportunity videos. The 22
caption-only videos reserved for fresh task construction were absent from the
returned table and remain unprompted, unscored, and unopened in Python.

## Frozen Gates

| Metric | Requirement | Result | Pass |
|---|---:|---:|:---:|
| Complete tasks | 40 | 38 | no |
| Tasks with at least 3 root top-1 clips | >=30 | 40 | yes |
| Tasks gaining a gold clip at depth two | >=15 | 27 | yes |
| Ordered bridge-to-answer chains recovered | >=10 | 21 | yes |
| Mean oracle pair coverage | >=.40 | .8125 | yes |
| Mean pair-coverage gain | >=.15 | .3375 | yes |
| Strict bridge opportunities | >=5 | 1 | no |
| Total strict oracle-over-greedy gap | >=5 | 1 | no |
| Mean strict direct-answer sacrifice | >=.15 | .05 | no |

Two tasks had one non-stopword answer token rather than the required two. That
schema miss is not decisive: the strict non-myopic gates fail by a wide margin.

## Mechanism

- Greedy and oracle roots are identical on `38/40` tasks.
- Oracle roots have lower direct-answer support on only `2/40`.
- The greedy root starts on the answer-side evidence clip on `22/40`.
- The oracle starts on the bridge clip on `16/40`.
- Yet the greedy root and its own continuation recover both clips on `23/40`.
- Only one task satisfies every frozen bridge-first condition, and its direct
  answer sacrifice is `.05`.

LongVidSearch does contain real sequential retrieval structure. Depth two gains
one necessary clip on 27 tasks, and 21 ordered chains are recoverable from
observation-only terms. The missing condition is action-order conflict: because
the task has exactly two evidence clips and the question remains visible, an
answer-side first search can often use its remaining search to backtrack to the
bridge. Thus the same first action is optimal both immediately and at the
two-search horizon.

## Consequence

No top-k, followup grammar, answer-overlap proxy, task subset, or threshold is
changed after this result. The exact two-hop construction is closed.

The source-defined three-hop family is a legitimate new prospective test:
three individually necessary clips and three searches leave less slack for
answer-side backtracking. It must use new caption-unopened official videos and
be frozen before any three-hop retrieval outcome. A pass would still authorize
only a 10-call serving smoke.

## Cost

- OpenRouter requests: `0`
- OpenRouter cost: `$0`
- OatML/Slurm jobs: `0`
- authenticated balance: `$33.574042594`
- Monday reserve: `$25`
- unused pre-Monday spend ceiling: `$8.50`
