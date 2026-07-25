# tau-Knowledge V3.1 Nonsemantic Control Plan

## Purpose

This is a post hoc, zero-call reviewer diagnostic on the frozen V3.1
confirmation trees. It tests whether simple target-blind retrieval heuristics can
replace the LLM's semantic continuation scorer. Definitions are fixed before
their endpoint totals are computed, but the main confirmation result and
endpoints are already known, so this cannot become a new preregistered claim.

## Shared Inputs

Every control uses the exact five roots, four followups, BM25 top-3 documents,
generated opening, initial information needs, and root-specific refreshed needs
stored in the public confirmation artifact. Required-document IDs are used only
after selections are frozen to score the endpoint.

All selections use deterministic original-order argmax tie breaking.

## Controls

### Raw BM25 Sum

For each root/followup pair, deduplicate the union of its first and followup
documents by document ID and sum their stored BM25 scores. Select the highest
followup beneath each root, then the root with the highest pair score.

This is intentionally simple despite query-dependent BM25 scales; it asks
whether raw retrieval confidence alone explains the result.

### Novel Document Count

For each pair, count distinct followup document IDs not already present in the
root's first results. Select the highest-count followup and then the root with
the highest resulting count.

### Lexical-IDF Overlap

Tokenize lowercase alphanumeric terms of length at least three, excluding a
frozen English stopword list. For each task, compute smoothed IDF over every
unique retrieved document in its complete tree:

`idf(t) = log((1 + N) / (1 + df(t))) + 1`.

For a root, the objective text is the opening, initial information needs, and
that root's refreshed information needs. A pair score is the sum, over unique
documents in the first/followup union, of the IDF weight of objective tokens
present in that document, divided by total objective-token IDF. Select the
highest pair.

This control consumes the LLM-generated beliefs but replaces semantic document
classification by literal lexical matching.

## Frozen Reporting Rule

Report endpoint totals, paired wins/losses/ties, mean differences, root-ranking
accuracy, continuation-ranking accuracy, optimal continuation rate, and regret
for all controls.

Call semantic scoring load-bearing in this diagnostic only if V3.1:

- exceeds every nonsemantic policy by at least three required documents;
- has more paired wins than losses against every policy; and
- has higher root and continuation pairwise accuracy than every policy.

Otherwise report which simpler component matches or beats it. No prompt,
heuristic, stopword, threshold, or endpoint may be changed after computation.
