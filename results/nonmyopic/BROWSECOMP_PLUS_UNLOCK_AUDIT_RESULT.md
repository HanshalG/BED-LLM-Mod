# BrowseComp-Plus Sequential-Retrieval Unlock Audit Result

## Decision

The frozen zero-call opportunity audit failed. The query-specific controlled
BrowseComp-Plus route is closed without model calls, threshold repair, task
replacement, or development/holdout access.

The failure is complete at the delayed-value link: none of 20 queries gains a
human evidence document from its best second retrieval over its best first
retrieval. There is therefore no utility for a non-myopic semantic scorer to
rank in this frozen tree.

## Frozen Results

| Metric | Result | Gate |
|---|---:|---:|
| Rows with at least three unique roots | 20/20 | at least 16 |
| Mean distinct first-result documents | 4.35 | at least 2.50 |
| Rows with a positive pair gain | 0/20 | at least 8 |
| Mean pair gain | 0.00 evidence documents | at least 0.40 |
| Oracle root differs from immediate root | 0/20 | at least 4 |
| Oracle beats immediate root plus oracle tail | 0/20 | at least 3 |
| Mean non-myopic gap | 0.00 evidence documents | at least 0.15 |
| Direct-query evidence coverage | 2/111 (1.80%) | below 60% |

The candidate pools contain 30 to 120 documents per query and the deterministic
roots retrieve 4.35 distinct top documents on average. The absence of pair gain
is not action collapse or direct-retrieval saturation.

Across all oracle pairs, six gold documents are retrieved, but gold-document
retrieval does not create the frozen primary evidence-coverage gain. Gold and
evidence qrels are separate benchmark judgments, and the primary endpoint is
unchanged after observing that secondary result.

## Interpretation

BrowseComp-Plus is a hard deep-research benchmark, not automatically a
non-myopic experimental-design benchmark. In this controlled candidate pool,
lexical roots almost never retrieve human evidence, and adding high-IDF terms
from a returned hard negative does not make another evidence document
reachable.

This result is specific to the preregistered query-specific corpus, BM25
transition, deterministic clause roots, and pseudo-relevance-feedback
continuations. It does not evaluate standard-corpus dense retrieval or
BrowseComp-Plus agents. Per protocol, those are not substituted after seeing
the endpoint.

The joint lesson from BRIGHT biology and BrowseComp-Plus is that retrieval
difficulty is not sufficient. A useful environment needs an externally
verifiable observation that unlocks a different action or support transition;
otherwise deeper planning only ranks zero-valued branches.

## Integrity and Cost

- Official code commit:
  `046949032b0328319cc9a02663a759ec601d9402`.
- Hugging Face dataset commit:
  `144cff8e35b5eaef7e526346aa60774a9deb941f`.
- Opportunity artifact SHA-256:
  `3f5f602ad2cc9b8777014cd9b9b942f5087c58ee2043f527a816bd3ec427d91e`.
- API requests: `0`.
- OpenRouter spend: `$0`.
- OatML use: none.
- Plaintext benchmark queries/documents committed: no.

The 20 development and 50 holdout rows remain unread.
