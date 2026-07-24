# MuSiQue Answer-Belief Bridge Result

Date: 2026-07-24
Seed: `24327`
Status: **opportunity conjunction failed; no planner or reserve evaluation**.

## Execution

The two-row serving smoke passed exactly 66 calls with zero reasoning tokens,
valid eight-answer beliefs throughout, and replay truth-probability gaps of
`.00` and `.02`. It cost `$0.28084875`.

The fresh six-row opportunity screen then completed exactly 198 calls:
192 GPT-5.4 belief transitions and six GPT-5.4 Mini equivalence judgments.
There were zero reasoning tokens, retries, forced exits, parser failures, or
runtime failures. It cost `$0.85151225`.

## Frozen gates

| Gate | Threshold | Result | Pass |
|---|---:|---:|:---:|
| Gold root proposed | at least 4/6 | 6/6 | yes |
| Gold continuation after root | at least 4/6 | 6/6 | yes |
| Gold pair is two-step oracle | at least 2/6 | 2/6 | yes |
| Oracle first differs from greedy | at least 2/6 | 4/6 | yes |
| Gold root differs from greedy | at least 2/6 | 5/6 | yes |
| Pair gain at least `.10` | at least 3/6 | 4/6 | yes |
| Mean pair gain | at least `.08` | `.2300` | yes |
| Non-myopic gap at least `.10` | at least 2/6 | 1/6 | **no** |
| Mean non-myopic gap | at least `.05` | `.0417` | **no** |
| Mean / max replay gap | at most `.10` / `.25` | `.0117` / `.0400` | yes |

Mean initial truth probability was `.0900`, below the frozen `.25`
saturation ceiling. All execution and remaining scientific gates passed.

## What failed

The LLM representation did exhibit useful path-dependent answer recovery.
Opening two documents raised truth probability by `.23` over the best single
document on average, the best first action changed on four of six rows, and the
gold ordered pair was the oracle on two rows.

That did not translate into enough value over the realized-myopic
continuation. The interface exposed the titles of all 20 documents before the
first choice. A myopic policy could therefore open the answer-bearing gold
document first and the bridge document second. Because the final belief then
received the same two texts as the gold order, the evidence was effectively
commutative. On one row, for example, the gold order reached truth probability
`1.0`, but the greedy reverse order also reached `1.0`.

This localizes the missing condition more tightly than the earlier chain-support
test: a semantic bridge is not sufficient if both hops are selectable at time
zero. The observation must create or identify the second action for order to
have irreducible value.

## Decision

Close this exact visible-title answer-belief interface. Do not run a planner,
reserve rows, threshold change, subset, or candidate-count repair.

A scientifically distinct successor may use a real open-domain retrieval
interface in which the corpus titles are hidden and the first retrieved passage
reveals the entity needed to formulate the second query. That design must receive
its own seed, opportunity gate, and preregistration.

Artifacts:

- `results/nonmyopic/musique_answer_belief_bridge_smoke/musique-answer-belief-bridge-smoke-20260724T231500Z/SERVING_SMOKE.json`
- `results/nonmyopic/musique_answer_belief_bridge_opportunity/musique-answer-belief-bridge-opportunity-20260724T232000Z/OPPORTUNITY.json`
