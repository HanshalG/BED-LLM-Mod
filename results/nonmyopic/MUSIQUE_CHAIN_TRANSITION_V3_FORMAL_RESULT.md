# MuSiQue Indexed Chain-Transition V3 Formal Result

Date: 2026-07-24
Run: `musique-chain-transition-v3-formal-20260724T201010Z`
Status: **scientific conjunction failed; no ranking or policy run**.

## Clean execution

The fresh 12-row formal gate completed exactly 84 requests: 12 initial generated
supports and 72 opened-document refreshes. There were zero reasoning tokens, retries,
forced exits, parser failures, or runtime failures. Cost was `$0.02360175`.

## Frozen gates

| Gate | Threshold | Result | Pass |
|---|---:|---:|:---:|
| True root among six actions | at least 8/12 | 11/12 | yes |
| Gold chain initially omitted | at least 4/12 | 8/12 | yes |
| Omitted chain recovered | at least 3 | 4 | yes |
| Mean oracle coverage gain | at least .15 | .333 | yes |
| Rows with branch spread | at least 4 | 4 | yes |
| Mean immediate-EIG regret | at least .10 | .083 | **no** |
| Rows with positive immediate-EIG regret | at least 3 | 1 | **no** |

All execution gates also passed. The frozen conjunction therefore fails.

## Mechanism

The LLM-generated representation is genuinely load-bearing: eight gold chains began
outside the generated support, and opening different stored documents caused four of
them to enter under one branch but not others. Mean oracle recovery gain was `.333`
(90% row bootstrap interval `[.083, .583]`).

That transition did not create enough non-myopic decision opportunity. Whenever the
true root appeared among the six actions, it occupied root slot 1 in 10/11 rows. The
fixed support multiplicities gave roots 1 and 2 the largest immediate EIG, with slot 1
as the deterministic tie break. Immediate EIG therefore selected 3/4 recovering
branches; seeded random selected 0/4. Immediate-EIG regret was only `.083` (90% interval
`[0, .25]`) and positive on one row, versus random regret `.333`.

The four initially covered chains also remained covered after every opened document.
Thus the refresh process shows both useful recovery and support inertia, but not a
reliable tradeoff that lookahead can exploit.

## Conclusion

MuSiQue guarantees connected multihop reasoning for final answer recovery, but this
does not imply a non-myopic experimental-design choice. Here the semantic generator
usually identified the useful first hop immediately. The exact indexed chain-support
route closes with no ranker, depth comparison, or post-hoc subset.

Artifact:
`results/nonmyopic/musique_chain_transition_v3_formal/musique-chain-transition-v3-formal-20260724T201010Z/GATE.json`.
