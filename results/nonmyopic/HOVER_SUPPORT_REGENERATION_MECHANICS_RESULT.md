# HoVer Support-Regeneration Mechanics Result

## Outcome

**Gate failed.** Serving and belief regeneration are clean, and several
mechanism diagnostics pass, but the regenerated non-myopic policy does not
change either myopic root or improve exact first-link value.

- Public result SHA-256:
  `4d17348332645f79ec3a825797177abd601b4c7b13bc3d30d5b6d96d7bdd2c76`.
- Private raw response SHA-256:
  `4a7c207171eb49cd02aaa499ab7e80cc5a0d0aced47149adc7d7f7ad30e1332e`.
- Exact physical requests / HTTP attempts: `28 / 28`.
- Transport retries: `0`.
- Reasoning tokens / forced exits: `0 / 0`.
- Responses parsed without repair: `28 / 28`.
- Cost: `$0.3514125`, below the frozen `$0.50` cap.
- OatML jobs: `0`.

## Passed Links

- All 20 regenerated weighted hypothesis states differ from their initial
  state.
- Each task has ten pairwise-distinct regenerated states.
- Aligned future score vectors are nonconstant and differ from both
  fixed-support and shuffled-future vectors on both tasks.
- The depth-3 oracle root proposes its exact first continuation on both tasks.
- Oracle-root regeneration increases exact support-title coverage on one task.
- Pooled aligned full-score pairwise accuracy against exact root values is
  `0.6087` over 46 comparable root pairs, above the frozen `0.55` gate.

These results show that GPT-5.4 can generate path-conditioned semantic beliefs
and sometimes recognize useful future structure. They do not survive the
argmax decision.

## Failed Decision Link

| Task | Myopic direct root score | Oracle-root direct score | Myopic future uplift | Oracle future uplift | Selected exact V3 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Perfect Match / Lauren London | 98 | 0 | 95 | 89 / 90 | 1 vs oracle 2 |
| The Stand / Warren Frost | 98 | 4 | 95 | 86 | 1 vs oracle 2 |

Adding aligned future uplift leaves the myopic root unchanged on both tasks.
Every evaluated policy except random selects mean exact `V3 = 1.0`; the oracle
value is 2 on both. Thus the frozen gates for root change, selected-value gain,
and oracle-root selection fail.

## Deeper Diagnosis

The post-response support audit reveals an environment mismatch:

- Task 1 initial generated support covers 2/3 exact support titles; every one
  of ten root-conditioned states covers 3/3.
- Task 2 initial generated support already covers 3/3; every root-conditioned
  state remains at 3/3.
- Root-conditioned support-coverage range is therefore zero on both tasks.

The claim and top-100 title catalog let a frontier model reconstruct the
evidence chain even after an irrelevant observation. The supposed valuable
setup root does not induce a uniquely better belief state.

There is also a transition mismatch. The zero-cost opportunity audit defines a
continuation as legal only when its title is mentioned in the opened article.
The LLM's natural open-domain policy proposes any semantically relevant
catalog title. For example, after opening `The Stand (miniseries)` it proposes
`Warren Frost` even though that title is not licensed by the audit's literal
mention graph. Under unrestricted semantic retrieval, the myopic root can
recover the same later support and retrieval becomes order-commutative. If the
literal graph rule were enforced, the task would become classical finite graph
navigation and the LLM's role would again be ornamental.

## Decision

Close this exact HoVer interface:

- no score rescaling or future-weight tuning;
- no legal-title menu or parser restriction;
- no delexicalization repair;
- no development or holdout calls.

HoVer contributes a useful caution: an exact structural lookahead gap can
disappear when the LLM is allowed to use its natural semantic retrieval
capability. The next route must use private or non-memorized observations and
must screen for root-dependent regenerated truth-support coverage, not only a
document-link graph.
