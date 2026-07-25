# BRIGHT Biology Sequential-Retrieval Unlock Audit Result

## Decision

The frozen zero-call opportunity audit failed. BRIGHT biology is closed without
model calls, threshold repair, another domain, or holdout access.

The corpus is difficult and the roots are diverse, but the returned documents
rarely make a relevant second retrieval reachable. This is the wrong failure
mode for a non-myopic scorer experiment: semantic ranking cannot recover value
that is absent from almost every generated branch.

## Frozen Results

The exact 20-row opportunity screen produced:

| Metric | Result | Gate |
|---|---:|---:|
| Rows with at least three unique roots | 20/20 | at least 16 |
| Mean distinct first-result documents | 4.30 | at least 2.50 |
| Rows with a positive pair gain | 3/20 | at least 10 |
| Mean pair gain | 0.15 gold chunks | at least 0.50 |
| Oracle root differs from immediate root | 2/20 | at least 5 |
| Oracle beats immediate root plus oracle tail | 2/20 | at least 4 |
| Mean non-myopic gap | 0.10 gold chunks | at least 0.20 |
| Direct-query gold coverage | 2/63 (3.17%) | below 60% |

Three gates pass: every row is analyzable, first-result diversity is high, and
direct retrieval is far from saturated. Five load-bearing opportunity gates
fail.

Only query IDs `52` and `30` have a positive non-myopic gap, each worth one
gold chunk. Query `18` gains a second gold chunk at depth two, but the
immediate root already exposes that best continuation, so it has no root-choice
gap.

## Interpretation

The failure is retrieval reachability, not scorer noise. The fixed roots return
4.3 different top documents per task, yet pseudo-relevance feedback reaches a
new gold chunk on only three tasks. A deeper semantic scorer would mostly rank
branches whose exact endpoint is zero.

This distinguishes BRIGHT biology from tau-Knowledge. BRIGHT was designed to
require reasoning-intensive matching, and its direct BM25 recall is low; those
facts make it a hard retrieval benchmark but do not by themselves create an
enabling first action. In the frozen tree, observing a plausible but irrelevant
document generally does not expose a lexical route to a gold document.

No claim is made about BRIGHT as a whole or about stronger dense retrievers.
The result closes only this preregistered biology/BM25/pseudo-relevance-feedback
route. Per protocol, no alternate BRIGHT domain is substituted after seeing
these endpoints.

## Integrity and Cost

- Official code commit:
  `d99e8391d967d4c2b3a74732530d2309e2fc92b6`.
- Hugging Face dataset commit:
  `3066d29c9651a576c8aba4832d249807b181ecae`.
- Opportunity artifact SHA-256:
  `d4769d2dea266337453555f2ea9169e0762a2ab24b90f5cabe443bd0b0fcfe47`.
- API requests: `0`.
- OpenRouter spend: `$0`.
- OatML use: none.

The 20 development and 50 holdout query IDs remain unused.
