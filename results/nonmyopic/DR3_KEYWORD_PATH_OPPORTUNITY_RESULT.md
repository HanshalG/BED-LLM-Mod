# DR3 Keyword-Path Opportunity Result

## Decision

The frozen zero-call opportunity gate fails and closes this exact construction
before any LLM use. DR3-Eval's static corpus supports useful sequential
retrieval, but hidden construction-keyword coverage is order-commutative under
the preregistered three-page actions. No serving smoke, development task, or
holdout task is authorized.

## Reproduction

- Official code commit:
  `86fed3760a8708d48121c4e9eaf0fddc939c6bef`.
- Official dataset revision:
  `4305f9129529d4510f485af6c997b69e1e85b88d`.
- Opportunity split: 20 task IDs under seed `24691`.
- Development tasks parsed: `0/8`.
- Holdout tasks parsed: `0/10`.
- OpenRouter requests/cost: `0 / $0`.
- OatML or cluster work: none.
- Public audit SHA-256:
  `ce72087c80854ab320fe94cabcb4b8aea3b306b829e588d14eb6b055490a55e7`.

## Frozen Results

| Metric | Result | Gate |
| --- | ---: | ---: |
| Completed tasks | `20/20` | `20/20` |
| Tasks with diverse root top-1 pages | `20/20` | at least `15` |
| Tasks gaining keywords at depth two | `20/20` | at least `10` |
| Mean best-immediate coverage | `.300` | descriptive |
| Mean oracle-pair coverage | `.590` | at least `.300` |
| Mean pair-coverage gain | `.290` | at least `.080` |
| Strict non-myopic opportunities | `0/20` | at least `4` |
| Strict total keyword gap | `0` | at least `4` |
| Mean strict normalized gap | `.000` | at least `.100` |

Every task's best root retrieved exactly three distinct keywords, the maximum
possible from three returned pages. Eighteen oracle pairs covered six keywords
and two covered five. All tasks therefore gained at depth two, usually by
three keywords.

Root-conditioned futures were not identical. Across tasks, root pair values
ranged from `3`--`5` at the low end to `5`--`6` at the high end, and roots had
an average of `10.7` distinct top-1 pages. Lower-immediate roots also existed:
11 tasks had root immediate values ranging from two to three keywords and nine
ranged from one to three.

The decisive failure is that every task had at least one maximum-final root
that also achieved maximum immediate coverage. Frozen greedy tie-breaking
therefore selected the same root as the two-step oracle on `20/20` tasks. No
root had to sacrifice immediate coverage to reach the best future.

## Interpretation

This is not evidence that semantic branch regeneration is useless. It is
evidence that this exact endpoint cannot distinguish myopic from non-myopic
selection. A policy can retrieve six useful facets in two searches, but an
immediately optimal first search always retains that option.

Changing retrieval width, weighting keywords, adding query costs, or redefining
value after observing the flat immediate ceiling would manufacture the desired
tradeoff. Those changes are forbidden by the preregistration. The public DR3
corpus may remain a supporting deep-research resource, but this construction
does not unlock paid work toward the headline claim.
