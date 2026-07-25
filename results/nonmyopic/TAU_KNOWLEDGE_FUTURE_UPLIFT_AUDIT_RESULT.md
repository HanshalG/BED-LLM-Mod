# tau-Knowledge Two-Split Future-Uplift Audit Result

## Decision

The audit passes every frozen strong-future-value criterion. Across two
disjoint 20-task blocks, the within-root score change caused by exposing the
full semantic future ranks the exact incremental required-document value of
that future.

This is the strongest first-link mechanism evidence in the project. It remains
post hoc analysis on open tasks and does not isolate regenerated belief text
from the future queries and documents generated with it.

## Reproduction

- First-link V2 confirmation SHA-256:
  `dfbf597f8405b109d61d90206606962b5fbda439c8b81f086a9592c07fa247d1`.
- Receding V3.1 confirmation SHA-256:
  `f8b120b0d2b98f9f10eb0ef9251262a6a41b20d4d90d724ee4926c141ec275ae`.
- Task-set overlap: `0/40`.
- Public analysis SHA-256:
  `773ed107b5cadd28d01d1fd03f301b46907b4d3f6112d15f45bcb51c0d08a680`.
- OpenRouter calls / cost / OatML use: `0 / $0 / none`.

## Results

| Metric | First-link V2 | Receding V3.1 | Pooled |
| --- | ---: | ---: | ---: |
| Tasks | `20` | `20` | `40` |
| Tasks with comparable future gain | `16` | `17` | `33` |
| Comparable root pairs | `89` | `98` | `187` |
| Uplift vs future-gain accuracy | `.6180` | `.6684` | `.6444` |
| Exact task sign-flip `p` | `.04382` | `.00941` | `.00120` |
| Full score vs future gain | `.5393` | `.6122` | `.5775` |
| Myopic score vs future gain | `.4438` | `.4796` | `.4626` |
| Myopic score vs immediate value | `.7745` | `.6768` | `.7264` |
| Full score vs total two-step value | `.7788` | `.7025` | `.7393` |

The algebraic links behave coherently:

- the isolated score ranks immediate retrieval value;
- the full-tree score ranks total two-step value; and
- `full - myopic` ranks incremental follow-up value.

## Selection Diagnostic

Selecting the maximum score uplift retrieves `43` units of exact incremental
future value across 40 tasks, versus `32` under myopic-score selection and
`35` under deployed full-tree-score selection. Uplift and deployed full-tree
roots agree on only `10/40` tasks.

This is not a newly proposed deployment policy: the uplift argmax was frozen
as a diagnostic after both endpoint sets were open. Its advantage identifies a
calibration opportunity for future prospective work, where immediate and
incremental semantic value should be estimated separately and combined on a
common scale.

## Frozen Gates

All strong conditions pass:

- first split comparable pairs `89 >= 30`;
- second split comparable pairs `98 >= 30`;
- split accuracies `.6180` and `.6684`, both above `.50`;
- pooled accuracy `.6444 >= .60`; and
- pooled exact task-level `p=.00120 <= .05`.

## Interpretation

The result rules out a simple explanation in which the non-myopic scorer is
merely a generically better root scorer. Its *increment relative to the
isolated myopic view* contains information about exact future-only document
gain on both task blocks. Full and myopic scores alone are substantially less
accurate for that incremental label.

The defensible causal chain is now:

1. LLM-generated semantic trees expose future retrieval consequences.
2. Exposing those futures changes scores in a direction that predicts exact
   incremental value.
3. Full-tree ranking improves total root-value accuracy on the held-out V3.1
   block.
4. Receding semantic continuation retains most selected-root value.

The remaining gap is narrower but important. The future tree bundles
regenerated hypotheses, generated queries, and retrieved documents. Earlier
belief-only shuffles were null/adverse, so this audit establishes semantic
future-value sensitivity, not causal dependence on correct refreshed-belief
text.
