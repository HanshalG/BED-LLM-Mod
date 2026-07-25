# tau-Knowledge Receding Continuation V3.1 Result

## Decision

The untouched 20-task confirmation passed every frozen gate. This is the first
held-out end-to-end result in the project where the LLM owns the path-dependent
semantic belief and retrieval machinery rather than merely proposing actions for
an exact finite verifier.

The result remains qualified. V3.1 was a transparent format-only amendment
chosen after iterative development, and the final exact-document advantages are
directional but underpowered. The root-ranking and continuation-ranking links
are statistically supported.

## Protocol and Serving

GPT-5.4 generated an opening, eight initial information-need hypotheses, five
root searches, refreshed hypotheses after each root's documents, and four
continuations per root. Required-document IDs and full user scripts were hidden
from every scorer.

The non-myopic root scorer saw each complete depth-two tree; the isolated myopic
scorer saw only first-result sets. At the realized second link, the same
count-dominant scorer was used beneath either selected root. It classified
distinct new useful documents under the initial and refreshed beliefs, with
refreshed beliefs explicitly treated as fallible.

Confirmation completed exactly 280 physical requests with:

- 20 generated openings;
- 20 initial hypothesis/root trees;
- 100 document-conditioned continuation trees;
- 20 isolated myopic root scores;
- 20 full-tree non-myopic root scores; and
- 100 focused receding continuation scores.

There were zero reasoning tokens, retries, forced exits, parse errors, or
runtime failures. Twenty-one focused responses used the amended unambiguous
zero-padding form. Cost was `$2.350365`.

## Structural Opportunity

The blind holdout had materially more non-myopic structure than the earlier
split:

- the oracle root differed from immediate greedy on 8/20 tasks;
- 6/20 tasks had a positive continuation-dependent root gap;
- the total gap was seven required documents, mean `0.35`;
- best one-step, greedy-continuation, and oracle-pair totals were 25, 37, and 44.

## Frozen Gates

Every preregistered confirmation condition passed:

| Component | Result | Gate |
|---|---:|---:|
| Root comparable pairs | 121 | >=50 |
| Myopic root accuracy | 0.5537 | control |
| Non-myopic root accuracy | 0.7025 | >=0.60 |
| Root accuracy gain | +0.1488 | >=+0.05 |
| Continuation comparable pairs | 228 | >=200 |
| Continuation accuracy | 0.7566 | >=0.60 |
| Oracle-optimal continuations | 80/100 | >=70/100 |
| Mean continuation regret | 0.21 | <=0.30 |
| Selected-root continuation loss | 3 | <=5 |

End-to-end required-document coverage was:

| Policy | Total | W/L/T versus non-myopic | Non-myopic gain |
|---|---:|---:|---:|
| Non-myopic receding | 30 | -- | -- |
| Myopic receding | 26 | 4/2/14 | +4 |
| Original joint non-myopic | 25 | 7/2/11 | +5 |
| Seeded random | 22 | 9/3/8 | +8 |

The selected non-myopic roots had oracle-tail value 33, so focused continuation
retained 30/33 documents and lost only three. The unrestricted pair oracle
reached 44.

## Paired Uncertainty

A deterministic zero-call analysis used exact task-level sign flips and 100,000
paired task bootstraps with seed `24339`.

- Root pairwise-accuracy gain: exact one-sided `p=0.03516`, bootstrap 95%
  interval `[0.0161, 0.2963]`.
- Focused continuation ranking above chance: exact one-sided
  `p=0.0000687`; accuracy interval `[0.6836, 0.8255]`.
- Root oracle-tail coverage was 33 versus 28 for myopic: exact `p=0.1875`,
  mean-gain interval `[-0.15, 0.65]`.
- End-to-end non-myopic versus myopic: mean `+0.20` documents, exact
  `p=0.25`, interval `[-0.20, 0.60]`.
- Versus original joint: mean `+0.25`, exact `p=0.08984`, interval
  `[-0.05, 0.50]`.
- Versus seeded random: mean `+0.40`, exact `p=0.07739`, interval
  `[-0.05, 0.85]`.

Thus both semantic ranking links are supported, and all frozen endpoint gates
pass, but 20 tasks do not establish a conventionally significant final coverage
gain.

## Interpretation

The positive mechanism is not generic deeper prompting. The full-tree scorer
values which first retrieval will induce useful future document branches.
After observing the selected root, the count-dominant scorer replans from the
LLM's regenerated information needs while preserving the original objective.
The LLM therefore performs the irreducible work: generating the semantic support,
transitioning it after evidence, and deciding which unstructured documents
support plausible unresolved needs.

The change from the earlier continuation failure is also measurable. The first
joint scorer lost ten of 44 selected-root oracle-tail documents. The final
receding scorer lost three of 33 on an independent split, ranked continuation
pairs at 0.7566 accuracy, and improved five documents over the original joint
selector.

This is held-out testing after iterative development, not a pristine one-shot
preregistration. V3's original strict parser rejected zero-padded strings even
though all semantic rows were complete. V3.1 explicitly superseded the
previous no-V4 promise before holdout access and accepted only one- or two-digit
strings in the unchanged count bands. No holdout response, endpoint, threshold,
or score was modified.

A separately frozen post hoc reviewer control used the identical trees but
replaced semantic scoring with raw BM25 sum, novel-document count, or
lexical-IDF overlap against the same generated beliefs. Those policies retrieved
22, 21, and 17 documents versus V3.1's 30; V3.1 also had higher root and
continuation accuracy than all three. This supports semantic scoring as
load-bearing, while remaining non-preregistered mechanism evidence.

## Artifacts and Budget

- Public confirmation SHA-256:
  `f8b120b0d2b98f9f10eb0ef9251262a6a41b20d4d90d724ee4926c141ec275ae`.
- Private raw SHA-256:
  `9448c769607dba39bfbac028a92663ea5cec0d7d740164cb66693f3f58b77cea`.
- Analysis SHA-256:
  `d119555cb718d7c0a7eef258e4cf56077b6cbf220ca48267169823b22fc548b3`.

The project ledger is `$78.315110` spent. OpenRouter has `$52.069693`
remaining, or `$27.069693` above the protected `$25` Monday reserve. OatML was
not used.
