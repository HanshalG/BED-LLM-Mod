# MuSiQue Adaptive-Retrieval Belief Result

Date: 2026-07-24
Seed: `24328`
Status: **serving/opportunity smoke failed; no six-row screen**.

## Clean execution

The full two-row 6x4 retrieval tree completed exactly 66 calls: 64 GPT-5.4
belief/query calls and two GPT-5.4 Mini equivalence calls. There were zero
reasoning tokens, retries, forced exits, parser failures, or runtime failures.
Every state produced a valid eight-answer belief. Cost was `$0.22363550`.

Both rows retrieved the gold first document and then the gold second document.
The gold ordered pair was the two-step truth-probability oracle on both rows.
Pair gain over the best one-step belief was `+.71` and `+.22`, and replay gaps
were `.14` and `.01`.

## Frozen failure

The smoke required at least two distinct first retrieved documents on each row.
The observed counts were three and one. On the second row, all six diverse
direct/bridge query strings retrieved the same first-hop document.

Consequently, realized greedy and the two-step oracle shared their first query
on both rows. The non-myopic gap over the greedy continuation was exactly zero
on both.

## Interpretation

Hiding document titles successfully made the second query observation-dependent,
and the model generated coherent bridge-conditioned follow-ups. But each
MuSiQue question still contains one deterministic reasoning chain. When every
sensible initial query retrieves its obvious root, the task tests iterative
retrieval rather than a choice among competing experiments.

The missing ingredient is now precise: multiple plausible latent branches must
share the initial problem, and the first observation must route them to different
useful continuation actions. A single gold chain is not enough.

## Decision

Close MuSiQue for this project. Do not run the frozen six-row opportunity screen,
resample query strings, tune BM25, relax the diversity gate, or expose additional
retrieval results.

Artifact:
`results/nonmyopic/musique_adaptive_retrieval_belief_smoke/musique-adaptive-retrieval-smoke-20260724T234000Z/SERVING_SMOKE.json`.
