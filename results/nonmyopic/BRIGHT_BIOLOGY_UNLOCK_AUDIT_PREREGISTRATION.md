# BRIGHT Biology Sequential-Retrieval Unlock Audit

## Status

Frozen before any BRIGHT endpoint calculation or model response. This is a
zero-call structural audit for a genuinely independent replication of the
tau-Knowledge mechanism. It does not authorize a scorer or policy experiment
unless every opportunity gate passes.

## Source and Split

- Benchmark: BRIGHT biology, CC-BY-4.0.
- Official code repository commit:
  `d99e8391d967d4c2b3a74732530d2309e2fc92b6`.
- Hugging Face dataset commit:
  `3066d29c9651a576c8aba4832d249807b181ecae`.
- Examples parquet SHA-256:
  `6e105c4f09d9a70b8a20ed6a4d0e386823a5545151df41b3f0e64eb5c5987829`.
- Documents parquet SHA-256:
  `8516d0c233f9c34e9eb6922b56e8a1698e5a6f6e504a9499fcd511cdd5741670`.
- Source rows `0` through `10` are excluded because they were visible during
  source and schema inspection.
- Selection seed: `24340`; numeric IDs `11` through `102` were shuffled before
  reading any endpoint.
- Serving smoke IDs: `96`, `78`.
- Opportunity IDs: `98`, `92`, `82`, `39`, `80`, `18`, `45`, `94`, `31`,
  `52`, `56`, `43`, `50`, `51`, `30`, `29`, `61`, `23`, `60`, `79`.
- Development IDs: `69`, `58`, `74`, `62`, `37`, `90`, `48`, `34`, `89`,
  `65`, `53`, `19`, `11`, `91`, `59`, `42`, `75`, `40`, `77`, `32`.
- Sealed holdout IDs: `44`, `102`, `46`, `76`, `36`, `14`, `67`, `41`,
  `13`, `24`, `16`, `88`, `20`, `84`, `21`, `15`, `97`, `72`, `73`, `87`,
  `64`, `54`, `99`, `12`, `71`, `22`, `66`, `100`, `57`, `47`, `35`, `83`,
  `55`, `101`, `33`, `68`, `93`, `49`, `81`, `26`, `70`, `95`, `85`, `17`,
  `25`, `28`, `38`, `27`, `63`, `86`.

The 50 holdout examples and their relevance labels remain unread unless a later
LLM-native development gate is separately frozen and passed.

## Deterministic Audit

The audit uses only the 20 opportunity rows and the fixed 57,359-document
biology corpus. Retrieval is BM25 over lowercase alphanumeric tokens. Retrieved
documents are excluded from later searches in the same branch.

Each query has five deterministic first searches:

1. the full raw query;
2. the first nonempty line;
3. the benchmark-provided reasoning text;
4. the final sentence ending in a question mark, or the final sentence; and
5. the longest remaining sentence.

Duplicate normalized searches are removed. A row is analyzable only with at
least three unique roots.

For every root, BM25 returns three documents. Four observation-conditioned
followups are then formed without relevance labels:

- one pseudo-relevance-feedback query for each returned document, combining
  the raw query with that document's 16 highest-IDF non-query terms; and
- one aggregate query combining the raw query with the eight highest-IDF
  non-query terms across all three returned documents.

Each followup retrieves three new documents. Utility is the number of distinct
official chunk-level `gold_ids` in the union of first and second retrievals.
The immediate control chooses the root with the most gold documents at step
one, with original root order breaking ties, then receives its oracle-best
followup. The non-myopic oracle chooses the best root-followup pair with the
same deterministic tie rule. This deliberately gives the immediate control an
oracle continuation; a positive gap therefore requires the first search itself
to matter.

## Gates

All conditions must pass:

- at least 16/20 rows have at least three unique roots;
- mean distinct first-result documents across roots is at least `2.5`;
- at least 10/20 rows gain at least one gold chunk from the best pair over the
  best first search;
- mean pair gain over the best first search is at least `0.50` gold chunks;
- the non-myopic root differs from the immediate root on at least 5/20 rows;
- the non-myopic oracle beats immediate-root-plus-oracle-continuation on at
  least 4/20 rows;
- mean non-myopic gap is at least `0.20` gold chunks; and
- original-query top-3 retrieval covers less than 60% of all opportunity gold
  chunks, preventing a saturated direct-retrieval task.

Failure closes BRIGHT biology without model calls, threshold repair, alternate
domain substitution, or holdout access. Passage authorizes only a separate
exact-10-call serving smoke and development preregistration transferring the
tau recipe: LLM-generated path-dependent needs and queries, fallible refresh,
target-blind semantic document classification, and count-dominant utility.

## Budget

This audit costs zero API calls. Any later serving smoke must check the live
OpenRouter balance, preserve the protected `$25` reserve, use non-reasoning
requests, fail closed, and cap projected smoke spend below `$0.50`. OatML
remains paused.
