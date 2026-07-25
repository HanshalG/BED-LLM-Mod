# MuSiQue 4-Hop Branching Unlock Audit Preregistration

Date frozen: 2026-07-25

## Purpose

Test whether official MuSiQue four-hop branching questions contain a
two-retrieval planning opportunity that survives the failure mode seen in the
two-hop MuSiQue and HotpotQA experiments.

The target `4hop3` composition has two independent roots:

1. a **deep root** whose answer is required to formulate an intermediate step;
2. that **deep child**;
3. a **shallow root**; and
4. a final step that combines the deep child and shallow root.

With two retrievals, deep-root first can resolve a connected two-step prefix.
Shallow-root first can retrieve two support documents but cannot resolve a
two-step dependency prefix. This audit tests only the external structural
opportunity and lexical temptation. It makes no LLM policy claim.

## Development Disclosure

The previously used MuSiQue dev split is not prospective. Before this
preregistration, aggregate statistics were inspected for all 95 dev `4hop3`
rows and eight examples were printed. Those observations defined this
protocol and its thresholds. No dev row will be presented as fresh evidence.

The source for this audit is the newly downloaded official answerable **train**
split. Before reading any train question, answer, decomposition, support
index, title, or paragraph, only the task IDs were used to freeze the split.

## Source And Frozen Split

- Official artifact: `musique_ans_v1.0_train.jsonl`.
- Source project: `https://github.com/StonyBrookNLP/musique`.
- Source SHA-256:
  `83a75b1e11e4e9bb8f8308e72ac40ca617ae4431b3a0d955b61cab259248490a`.
- Total rows: 19,938.
- Rows whose ID begins `4hop3__`: 400.
- Selection seed: `24353`.

Sorted eligible IDs were shuffled with `random.Random(24353)` and split:

- opportunity: first 120 IDs, ordered-list SHA-256
  `d11fbc929ebe04ca01e6fc88b7fe417e970a4ca5912b746a782ae51a3ed16423`;
- development: next 40 IDs, SHA-256
  `619615104c1357f08b51b6363c225ded64313035899626d7d98c1a39117ff8ee`;
- holdout: remaining 240 IDs, SHA-256
  `175a7dc642b88e843cbcfab8ae3a0ec44e036e93dcc40f9fcc18a3c56030d7b6`.

This audit may access endpoints only for the 120 opportunity IDs.

## Frozen Structural Rule

Parse each decomposition question for `#N` dependencies. A valid branching
row must have exactly four decomposition steps with dependency sets:

```text
step 1: {}
step 2: {1}
step 3: {}
step 4: {2, 3}
```

The four annotated support paragraph indices must be distinct and belong to a
20-paragraph context.

For the two-action structural endpoint:

- deep-first opens support steps 1 then 2, resolving one dependency edge and a
  connected prefix of length 2;
- shallow-first opens support steps 3 then 1, the best two-root fallback, but
  resolves no dependency edge and has maximum connected prefix length 1.

The frozen structural gap is therefore one connected step. This is an
evaluation endpoint, not a legality restriction: a later LLM smoke must
generate its own queries without seeing decomposition annotations.

## Lexical Baseline

Run deterministic BM25 over each row's 20 title-plus-paragraph documents using
the final question as query. Original paragraph order breaks ties.

Record:

- whether the shallow-root support scores above the deep-root support;
- whether the deep root is not rank one;
- whether rank one is the shallow root or a distractor;
- ranks and roles for all four support documents.

Also record whether either root answer or title occurs as a contiguous
normalized token sequence in the final question. No answer or annotation is
given to BM25.

## Frozen Gates

The zero-call audit passes only if all hold:

- source hash, total row count, eligible count, and all split hashes reproduce;
- exactly 120 opportunity rows are processed;
- all 120 have the exact branching dependency graph, four distinct supports,
  and 20 paragraphs;
- the structural deep-first minus shallow-first connected-prefix gap is exactly
  one on every row;
- neither root answer is stated in the final question on at least 110/120 rows;
- shallow-root BM25 score exceeds deep-root score on at least 45/120 rows;
- the deep root is not BM25 rank one on at least 85/120 rows;
- BM25 rank one is shallow-root or distractor on at least 65/120 rows;
- zero development or holdout endpoints and zero model calls are accessed.

Failure closes this higher-hop construction. Passing authorizes only a
separately preregistered, small causal/query-generation smoke on the 40
development rows. The 240-row holdout remains sealed.

## Intended LLM-Native Follow-Up

The LLM would generate its own natural-language dependency-chain hypotheses
and retrieval queries from the final question, refresh them after a retrieved
paragraph, and predict continuation value. Hidden decompositions would be used
only for exact support-role and connected-prefix endpoints.

The key comparison would be a receding myopic semantic policy against a
model-aware depth-two policy on shared generated roots and retrieved
paragraphs. A valid result must show that the LLM-generated belief state
identifies the deep root before annotations are revealed; hard-coding the
decomposition or restricting actions to gold links is not allowed.

## Cost And Scheduling

- Opportunity audit: zero OpenRouter calls.
- Weekend paid-work target remains at most `$15`; `$25` is protected.
- OatML is paused and no cluster job is involved.
