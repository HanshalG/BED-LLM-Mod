# MuSiQue Adaptive-Retrieval Belief Gate Preregistration

Date: 2026-07-24
Seed: `24328`
Status: preregistered before any response.

## Motivation

The visible-title answer-belief gate recovered useful two-document beliefs but
failed the non-myopic conjunction because both gold documents were selectable at
time zero. Greedy could acquire the same evidence in reverse order.

This successor changes the environment, not a threshold or prompt detail.
Actions are now free-form search queries against a sealed corpus. No document
title or ID is visible before retrieval. The first search result can therefore
introduce an entity that did not exist in the initial action vocabulary and is
needed to formulate the second query.

A response-free audit on the frozen 14-row draw used the exact deterministic
BM25 backend:

- the answer passage was top-1 from the original question on only 1/14 rows;
- the first support passage was top-3 on 10/14 rows;
- after substituting the stored bridge answer into the second decomposition
  query, the answer passage was top-1 on 12/14 rows.

These audit queries are dataset diagnostics only and are never shown to a model
or used as candidate actions.

## Frozen design

- Hash-verified MuSiQue answerable dev v1.0.
- Fresh target-blind draw seed `24328`, excluding every chain and visible-menu
  row, including all reserves.
- Smoke: first two rows; opportunity screen: next six; final six remain sealed.
- GPT-5.4 generates answer beliefs and queries with no reasoning, temperature 0.
- GPT-5.4 Mini judges answer equivalence only after all beliefs freeze, also with
  no reasoning.
- Each row's 20 documents receive shuffled opaque IDs.
- Retrieval is deterministic BM25 over title plus text; ties use opaque-ID order.
- The initial prompt exposes only the question and asks for an eight-answer
  probability belief, three direct queries, and three bridge queries.
- Candidate order is deterministically shuffled.
- Each query returns one title and paragraph. The LLM then regenerates its belief
  and proposes four distinct observation-conditioned follow-up queries.
- Follow-up retrieval excludes the already opened document.
- Every ordered pair receives a fresh final belief; the first pair is replayed.
- Gold answers and support labels are hidden from the generator.
- Private raw responses are phase-checkpointed.

The complete 6x4 tree costs exactly 33 requests per row: one initial, six first
branches, 24 final branches, one replay, and one equivalence call. Smoke is 66
calls; opportunity is 198.

## Frozen gates

Smoke must complete all 66 calls with zero reasoning, valid eight-answer beliefs,
finite replay measurements, and at least two distinct first retrieved documents
per row.

The six-row opportunity conjunction requires:

1. exact 198 calls and zero reasoning;
2. mean initial truth probability at most `.25`;
3. mean distinct first retrievals at least `3.0`;
4. at least 4/6 rows retrieve the gold root under some first query;
5. at least 4/6 retrieve the gold second document after a gold-root result;
6. the gold ordered retrieval pair is the two-step oracle on at least 2/6;
7. the oracle first query differs from realized greedy on at least 2/6;
8. pair gain over best one-step truth probability is at least `.10` on 3/6 and
   averages at least `.10`;
9. oracle gain over the realized-greedy continuation is at least `.10` on 2/6
   and averages at least `.07`;
10. replay truth-probability gap has mean at most `.10` and maximum at most
    `.25`.

Failure closes this exact adaptive-retrieval opportunity design. No planner,
reserve evaluation, row subset, retrieval tuning, query repair, or threshold
change follows.

## Budget

- Smoke projected/hard cap: `$0.50` / `$1.50`.
- Opportunity projected/hard cap: `$1.50` / `$4.00`.
- Live balance before implementation was `$62.087730786`; project headroom above
  the protected Monday reserve was `$37.040571034`.
