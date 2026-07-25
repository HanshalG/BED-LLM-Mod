# BrowseComp-Plus Sequential-Retrieval Unlock Audit

## Status

Frozen before downloading, decrypting, or reading any BrowseComp-Plus query.
This is a zero-call upper-bound screen for an independent transfer of the
tau-Knowledge mechanism. A pass authorizes only an exact-10-call serving smoke;
a failure closes this route without model use.

## Source and Blind Split

- Official repository:
  `texttron/BrowseComp-Plus@046949032b0328319cc9a02663a759ec601d9402`.
- Hugging Face query/annotation dataset:
  `Tevatron/browsecomp-plus@144cff8e35b5eaef7e526346aa60774a9deb941f`.
- Evidence qrel SHA-256:
  `a6f594975be57339de9e4e9f67f13c044f647feda77c0b84c45a1581e3041bd1`.
- Gold qrel SHA-256:
  `b875af4a745712bee7a94f464ed989232f8c77977c31824428470e11dcb28c73`.
- Selection seed: `24341`. The 830 public query IDs were sorted numerically
  and shuffled before any query text or annotation row was read.
- Serving smoke IDs: `37`, `678`.
- Opportunity IDs: `873`, `291`, `223`, `1124`, `1107`, `1036`, `625`,
  `1010`, `1232`, `1003`, `992`, `1249`, `78`, `926`, `120`, `971`, `1149`,
  `181`, `367`, `357`.
- Development IDs: `734`, `991`, `1039`, `814`, `770`, `170`, `611`,
  `1065`, `1004`, `773`, `254`, `1253`, `1047`, `417`, `555`, `1209`, `916`,
  `532`, `716`, `800`.
- Holdout IDs: `715`, `896`, `1254`, `1257`, `93`, `3`, `286`, `865`,
  `1225`, `1076`, `735`, `781`, `1091`, `124`, `178`, `981`, `1220`, `276`,
  `335`, `664`, `169`, `850`, `1153`, `377`, `895`, `1127`, `96`, `706`,
  `128`, `202`, `1224`, `500`, `1057`, `241`, `538`, `215`, `1147`, `519`,
  `985`, `102`, `651`, `936`, `390`, `1028`, `1204`, `943`, `140`, `793`,
  `961`, `624`.

The development and holdout rows remain unread during this audit.

## Controlled Retrieval Task

For each opportunity query, its candidate corpus is the deduplicated union of
the benchmark-provided human evidence documents, gold documents, and mined hard
negatives. This is deliberately a query-specific controlled comparison set,
not the standard 100,195-document leaderboard corpus. Candidate construction
uses labels, but neither root nor followup generation receives document class.
Every policy later compared on this route would receive the identical candidate
set and BM25 transition.

Retrieval uses BM25 over lowercase alphanumeric tokens and returns the top
three documents. Documents already returned in a branch are excluded from its
second retrieval.

Each query receives five deterministic first searches:

1. the full query;
2. up to four longest unique clauses split at sentence boundaries, semicolons,
   commas, or the standalone connectors `and`, `while`, `whereas`, `which`,
   `who`, `whose`, and `that`;
3. if fewer than five unique roots result, the first and second token halves
   are appended in that order.

Clauses shorter than four tokens are omitted. Original position breaks
equal-length ties.

For every root, four observation-conditioned followups are formed without
labels: one query per returned document using the raw query plus that
document's 16 highest-IDF non-query terms, and one aggregate query using the
raw query plus the eight highest-IDF terms across all three documents.

Primary utility is the number of distinct human evidence document IDs retrieved
across the two searches. Gold-document coverage is recorded as a secondary
endpoint. The immediate control selects the root with maximum first-search
evidence coverage and then receives its oracle-best followup. The non-myopic
oracle selects the best root-followup pair. Original root/followup order breaks
ties. This oracle-strength immediate control means any positive gap must come
from first-search choice.

## Gates

All conditions must pass on the 20 opportunity rows:

- at least 16 rows have at least three unique roots;
- mean distinct top-1 document count across roots is at least `2.5`;
- best pair improves over best first search on at least 8 rows;
- mean pair gain is at least `0.40` evidence documents;
- oracle and immediate roots differ on at least 4 rows;
- oracle beats immediate-root-plus-oracle-tail on at least 3 rows;
- mean non-myopic gap is at least `0.15` evidence documents; and
- the full-query top-3 search covers less than 60% of all evidence documents.

Failure closes the query-specific BrowseComp-Plus route without threshold
repair, task replacement, model calls, or development/holdout access. Passage
authorizes a separate exact-10-call smoke on IDs `37` and `678`: GPT-5.4
non-reasoning generates four initial semantic searches and one refreshed
four-followup plan after each root, using the same hidden labels and retrieval
transition. No scorer or policy confirmation is authorized by this audit alone.

## Budget

The audit makes zero API requests. Any serving smoke must first check live
OpenRouter credits, preserve the protected `$25` reserve, project below
`$0.25`, cap at `$0.50`, fail closed, and leave OatML paused.
