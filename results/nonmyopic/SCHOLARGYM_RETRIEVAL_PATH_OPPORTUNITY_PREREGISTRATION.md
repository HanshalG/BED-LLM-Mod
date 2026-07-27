# ScholarGym Retrieval-Path Opportunity Preregistration

## Status

Frozen before computing any retrieval outcome. This is a zero-call structural
gate for an LLM-native sequential BED construction. A full pass authorizes only
a separately preregistered OpenRouter serving smoke and development mechanics
test. It is not itself a policy result.

## Source

- Official code: `shenhao-stu/ScholarGym` at commit
  `cb1c5fc7c796308bef353b18549a92e2739f5b96`.
- Official Hugging Face dataset revision:
  `be7d917ddac3cd6f2878f81160965f914dab3706`.
- Query benchmark SHA-256:
  `869f507eacb7f554b8f4e6dc65ea97b01f95140ba5aa86a5119b330c41b9d551`.
- Paper corpus SHA-256:
  `e1e570321e22bf59784d46a9dd28239350d7ad1ddda213eee2f5730710f239e8`.

The release has 2,536 valid queries from PaSa AutoScholar, PaSa RealScholar,
and LitSearch, with 4,498 unique ground-truth arXiv papers. The released corpus
has 570,206 keyed paper records; all 4,498 ground-truth IDs are present. The
README says 570,205 papers, so this audit records and validates the file count
rather than silently adopting the prose count.

ScholarGym describes Test-Hard and Test-Fast aggregates but does not release
their query-ID lists. This construction does not reconstruct or use those
outcome-selected subsets.

## Quarantine And Split

`AutoScholarQuery_test_0` through `AutoScholarQuery_test_99` are mechanics-only
because examples or dataset previews exposed records from that range during
source inspection. They can never enter opportunity, development, holdout, or
policy evidence.

Only queries with at least two distinct canonical ground-truth arXiv IDs are
eligible. Version suffixes are removed before deduplication. Selection preserves
released file order before a seeded shuffle:

- opportunity pool: 541 `AutoScholarQuery_dev_*` queries, seed `270727`;
  first 40 IDs have hash
  `6f930b3cd8a98fa565371150a87ddbb07143591a14f437379bf1d0aa9b5dbb53`;
- development pool: 480 `AutoScholarQuery_test_100..999` queries, seed
  `270728`; first 24 IDs have hash
  `dcf14914b3516afe2362915422b99c44beeae16ea4596778dad31f682ac6efdf`;
- holdout pool: 49 multi-paper `RealScholarQuery_*` queries, seed `270729`;
  first 24 IDs have hash
  `04fd9f531470f94f42d89f56cbc1f676ffa28466e6768ca617a140bbc896f15e`.

The corresponding complete shuffled-pool hashes are:

- opportunity:
  `9450c76b17fc77c499ee33659a87ef9cb09becc30fc87df76f0674112880fa4a`;
- development:
  `79895a164dc7a0d880a442e68097b834f1a99e2d1ea01fbb6277f5e5b68155ff`;
- holdout:
  `34305e45bb625d0ce8d2f8afc9da030a79d50a1c16f615cc5acba891de2c3916`.

The opportunity audit may inspect only the 40 opportunity queries and their
retrievals. Development and holdout bytes may be hash-verified and used to
reproduce ID selection, but their query text, paper labels, retrieval outcomes,
and task records remain sealed.

Protocol correction before outcome access: the initial frozen commit counted
citation-list entries and therefore reported 542 eligible development rows.
The first audit attempt stopped during split validation, before corpus indexing
or retrieval, because `AutoScholarQuery_dev_496` repeats arXiv ID `2304.07327`.
This revision freezes the intended distinct-canonical-ID rule and its corrected
541-row pool and hashes.

A second incomplete attempt was interrupted during the first opportunity task
after revealing that calling FTS5's scalar `bm25()` function defeated its
top-N optimization for broad OR queries. No record or aggregate was written or
inspected. The implementation now orders by FTS5's default `rank` column,
which is the documented auxiliary path for the same default BM25 score and
stable row-ID tie break. This is an execution-only correction; the retriever,
scores, candidates, and protocol are unchanged.

## Exact Environment

The visible initial state is the released research query. A search action is a
free-text query. The deterministic observation is the title and first 2,000
abstract characters for each of the top five papers. Search respects the
released query date constraint and strips arXiv version suffixes for matching.

The paper database is indexed locally with SQLite FTS5 using its `unicode61`
tokenizer and one indexed field containing title plus abstract. A search is an
OR query over unique lowercase alphabetic tokens. Results are ordered by FTS5
BM25 score and then corpus row ID. The second search excludes papers returned
by the first. This is a frozen deterministic sparse retriever, not a claim of
bitwise equivalence to ScholarGym's Python `rank_bm25` implementation.

History utility is exact ground-truth-paper recall. Immediate utility is recall
after the first five papers; pair utility is recall in the union of the first
and second sets of five papers.

## Target-Blind Search Tree

Each task receives at most 16 deterministic roots generated from:

- the full research query;
- its sentences and substantive clauses;
- its highest-IDF visible query terms; and
- two- to four-token query windows containing those terms.

Each root observation produces at most 24 followups from high-IDF terms found
only in visible titles and abstracts. Every followup must contain at least one
term absent from both the initial query and root query. Ground-truth paper IDs,
titles from unretrieved papers, development records, and holdout records are
never available to candidate generation.

For every root, the audit exhausts this fixed followup bank. Ties use frozen
candidate order.

## Strict Opportunity

- Greedy root: maximum immediate recall, then maximum best pair recall, then
  root order.
- Oracle two-step root: maximum best pair recall, then maximum immediate
  recall, then root order.

A strict non-myopic opportunity requires:

1. the roots differ;
2. the oracle root has strictly lower immediate recall;
3. its best pair has strictly higher recall than the greedy root and the
   greedy root's own best observation-conditioned continuation;
4. the oracle continuation adds at least one new ground-truth paper; and
5. the continuation contains an observation-derived term absent from the
   initial and root queries.

## Frozen Gates

All conditions must pass:

- all 40 opportunity tasks complete;
- every task has at least two ground-truth papers and five roots;
- at least 30 tasks have at least three distinct root top-1 papers;
- at least 15 tasks improve their best immediate recall at depth two;
- mean oracle pair recall is at least `.15`;
- mean pair-recall gain over best immediate recall is at least `.03`;
- at least 5 tasks meet the strict opportunity definition;
- strict tasks have a total non-myopic gap of at least 5 ground-truth papers;
  and
- mean normalized gap among strict tasks is at least `.15`.

No split, corpus, retriever, tokenization, root width, retrieval width,
followup width, endpoint, tie break, or threshold changes after outcomes.
Failure closes this exact construction before model use.

## Conditional LLM-Native Study

Only a full structural pass may authorize a fresh protocol. The intended
comparison is:

- a nonreasoning LLM proposes semantic information needs and root searches;
- each root observation causes the LLM to regenerate a path-dependent belief
  over missing papers/facets and a continuation search;
- a full-tree semantic scorer selects the first action;
- an isolated myopic scorer sees only immediate root observations;
- lexical, fixed-support, and seeded-random policies share the same two-search
  and retrieval budgets; and
- exact ground-truth-paper recall is the paired endpoint.

The LLM's irreducible role is semantic support formation and
observation-conditioned continuation, not a reasoning preamble. Reasoning is
reserved for the naive baseline.

The serving smoke must precede science, use OpenRouter only, and cost at most
`$0.20`. A development mechanics test must be separately frozen and capped at
`$0.80`. With the authenticated balance at `$33.57` on 2026-07-27, at least
`$25` remains reserved through Monday and total new pre-Monday spend is capped
at `$8.50`.
