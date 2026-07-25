# GSM-Agent Non-Myopic Retrieval Opportunity Preregistration

Date frozen: 2026-07-25

## Purpose

Test whether the official GSM-Agent search environment contains a strict,
externally enforced two-query opportunity suitable for LLM-native non-myopic
BED. A useful root query may retrieve fewer required documents immediately but
expose identifiers, language, or metadata that enable a better second query.

This audit is zero-call opportunity measurement only. It makes no LLM-policy,
belief-quality, or answer-accuracy claim.

## Official Source And Frozen Split

- Official repository:
  `https://github.com/GuoTianYu2000/GSM-Agent`.
- Frozen commit:
  `a596464ea79ae0b8b84830d1c78a7d065177b0e8`.
- Paper: Zhu et al., *GSM-Agent: Understanding Agentic Reasoning Using
  Controllable Environments*, ICLR 2026.
- Official full database JSON:
  `data_construction/full_database/data/11_GSM_final_database/final_database_full.json`.
- Source SHA-256:
  `948e1ad488ef5e7bc1ad9d605684441e77cf796ef923639d0c5414ae4fe3778c`.
- Source counts: 7,323 problems, 32,315 unique documents, and 1,073 official
  test problems.
- Selection seed: `24363`.

Before any official test question, premise, document, original solution, or
answer was inspected, the 1,073 test IDs were sorted and shuffled with Python
`random.Random(24363)`:

- opportunity: first 500 IDs, ordered-list SHA-256
  `852a1ce0796adac76188152a2f17ce31377fadc433a904ba3898c5e3bde764d7`;
- development: next 100 IDs, SHA-256
  `8c9830fa5d114decc160f63e87eddf7aafa8e7ae3cfb3095ac7b779845fc6543`;
- sealed holdout: remaining 473 IDs, SHA-256
  `79e8b4b40f3c77d09f3ffcd2ada5750d3188f47e94c7b7ab5bd738bed5d4c5a5`.

The complete shuffled test-ID hash is
`3245e5f8813d5871e02a5f2ce397104b96e7ae5d270ff1a1ffc7ed15404366cb`.
One training example was inspected to understand the schema. No test endpoint
was inspected before this protocol and split were frozen.

## Frozen Retrieval Protocol

The audit uses a dependency-free BM25 index over the content of all 32,315
official documents. Ties follow official corpus order. Each query returns five
documents, matching GSM-Agent's first-page budget.

Root candidates are generated only from the visible question:

1. the full question;
2. contiguous 2-, 3-, and 4-token windows after a fixed stopword filter;
3. pairs and individual tokens among the six highest-IDF visible question
   tokens.

Candidates are normalized, deduplicated in that order, and capped at 24.

For each root page, continuation candidates may use only the question and the
five returned documents' visible IDs, content, and metadata. They comprise
single and paired high-IDF newly visible terms, question-plus-term queries, and
document-ID queries, deduplicated and capped at 30. Labels, required-document
IDs, premises, and answers never enter candidate generation or retrieval.

Immediate utility is the number of required documents on the root page.
Two-step utility is required-document coverage in the union of root and
continuation pages. The greedy root maximizes immediate utility. The non-myopic
root maximizes the best attainable two-step utility, then immediate utility.
Original candidate order breaks all remaining ties.

## Strict Opportunity

A task is a strict non-myopic opportunity only if:

1. greedy and non-myopic select different roots;
2. the non-myopic root has strictly lower immediate required-document
   coverage;
3. after both roots receive their own best visible-history continuation, the
   non-myopic root has strictly higher two-step coverage; and
4. the non-myopic continuation discovers at least one required document not
   present on its root page.

Condition 3 is load-bearing. It rules out the failure seen in HotpotQA and
MuSiQue, where a strong receding continuation simply repaired the greedy root.

## Gates

The opportunity audit passes only if all hold:

- source hash/counts and all split sizes/hashes reproduce;
- exactly 500 opportunity tasks are processed, all from the official test
  split, without accessing development or holdout endpoints;
- at least 40/500 are strict non-myopic opportunities;
- at least 25 strict opportunities require at least four oracle documents;
- mean non-myopic two-step coverage gain over all 500 tasks is at least 0.08;
- every strict row has immediate sacrifice at least one and final coverage
  gain at least one.

Failure closes GSM-Agent before any model call. Passing authorizes only a
separately frozen, at-most-$0.50 LLM-native smoke on development tasks.

## Intended LLM-Native Stage If Authorized

The LLM will generate natural-language missing-premise hypotheses and root
queries, refresh those hypotheses from retrieved documents, and score
continuations over its own generated belief dynamics. Required-document
coverage and final numeric answers remain externally verifiable. Controls will
share generated roots and retrieval results:

- greedy immediate semantic recovery;
- fixed-support depth two;
- model-aware depth two with generated belief refresh;
- seeded random root;
- deterministic lexical oracle from this audit.

The deterministic lexical protocol is an audit oracle and runnable baseline,
not the proposed policy. The LLM remains load-bearing in open-vocabulary query
generation and semantic belief/transition scoring over a 32,315-document
environment.

## Cost And Scheduling

- Opportunity audit: zero OpenRouter calls and no cluster work.
- OatML is paused.
- Authenticated OpenRouter balance before implementation: `$44.710790384`.
- Preserve `$25`; cap all new spending through Monday 2026-07-27 at `$15`.
- No paid GSM-Agent stage is authorized by this document alone.
