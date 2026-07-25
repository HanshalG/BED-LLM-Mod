# 2WikiMultiHopQA Directional Chain Audit Preregistration

Date frozen: 2026-07-25

## Purpose

Test whether 2WikiMultiHopQA inference and compositional questions contain a
natural semantic setup action that is valuable only through the next retrieval
step. The candidate environment reveals one hidden paragraph per action. A
root paragraph can expose the bridge entity naming the answer-bearing child
paragraph; opening the child without the root does not expose the reverse
transition.

This audit tests opportunity only. It makes no LLM, policy, or answer-accuracy
claim and authorizes no paid call by itself.

## Official Source And Frozen Split

- Official repository: `https://github.com/Alab-NII/2wikimultihop`.
- Repository commit inspected:
  `13800e5be57df1b4040b9b1588c6c811779e69e9`.
- Paper: Ho et al., *Constructing A Multi-hop QA Dataset for Comprehensive
  Evaluation of Reasoning Steps*, COLING 2020,
  `https://aclanthology.org/2020.coling-main.580/`.
- Official April 7, 2021 `data_ids` archive:
  `https://www.dropbox.com/s/ms2m13252h6xubs/data_ids_april7.zip`.
- Archive SHA-256:
  `95df2bf56fdabe034e27aebc580e02264232203cf52552f9efe8a919e5529eef`.
- Extracted `train.json` SHA-256:
  `b318dbafbfed51a8029718fa59be8b616600cbff675a3b587694b28c5eedfc13`.
- Train metadata counts: 76,481 compositional, 4,379 inference, 51,963
  comparison, and 34,631 bridge-comparison rows.
- Eligible universe: the 80,860 inference and compositional IDs.
- Selection seed: `24360`.

Before any question, answer, context, supporting-fact, or evidence value was
read, eligible `(_id, type)` pairs were sorted by ID and shuffled with Python
`random.Random(24360)`. The ordered IDs were frozen as:

- opportunity: first 500 IDs, ordered-list SHA-256
  `a549809258236c603e1abd11f52409fd0e0a75ff5077bd798ca61d9ad4466fb2`
  (471 compositional, 29 inference);
- development: next 100 IDs, SHA-256
  `4b8ad7e698f8b1201cfebb0f7c663cd0d55c84fd368f4dff9a32f63cb04667a1`
  (94 compositional, 6 inference);
- sealed holdout: next 1,000 IDs, SHA-256
  `dedde590777821a56805d83d486e10975f1a733c8ee8aadeb56fb192bae11d58`
  (964 compositional, 36 inference).

This audit may access endpoint fields only for the 500 opportunity IDs.
Development and holdout endpoint fields remain sealed.

## Frozen Chain Rule

Normalize text with Unicode NFKD, ASCII case-folding, underscore replacement,
alphanumeric tokenization, and whitespace collapse. For titles, also admit the
normalized prefix before a final parenthetical disambiguator. Matching is by
contiguous normalized tokens, never substrings.

For each opportunity row:

1. Require type `inference` or `compositional`, exactly two evidence triples,
   and exactly two distinct supporting-fact titles that each occur once in the
   ten-document context.
2. Require the evidence triples to form an ordered chain:
   `(root_entity, relation_1, bridge_entity)` followed by
   `(bridge_entity, relation_2, answer_entity)`.
3. Map `root_entity` and `bridge_entity` uniquely to the two support titles by
   normalized title aliases. These are the **setup root** and **answer child**.
4. Require the setup-root paragraph to mention the answer-child title or bridge
   entity, and require the answer-child paragraph not to mention the setup-root
   title.
5. Exclude `yes`/`no`, answers shorter than three normalized characters, and
   rows where the normalized answer appears in annotated supporting sentences
   from zero or both support documents. The answer must occur only in the
   answer child's supporting sentences.

Under this exact directional chain, setup-root first followed by its exposed
child retrieves both annotated support documents in two actions. Starting from
the answer child retrieves at most one because the reverse transition is
absent.

## Frozen Shallow Controls

Record two deterministic root baselines, with original context order breaking
ties:

- **title BM25**: question query against the ten normalized context titles;
- **paragraph BM25**: question query against each normalized title plus full
  paragraph text.

Also record whether any normalized setup-root title alias occurs verbatim in
the question. These controls receive no answer, evidence, or support labels.

## Gates

The zero-call opportunity audit passes only if all hold:

- source hashes, eligible count, metadata counts, split sizes, type counts, and
  all three split hashes reproduce;
- exactly 500 opportunity rows are processed and no development or holdout
  endpoint field is accessed;
- at least 450/500 rows have the exact two-triple evidence chain;
- at least 300/500 rows satisfy the strict directional-chain rule;
- every strict chain has setup-first minus child-first two-action support
  coverage exactly one;
- at least 50 strict chains do not state any setup-root title alias in the
  question;
- title BM25 fails to rank the setup root first on at least 75 strict chains;
- paragraph BM25 fails to rank the setup root first on at least 75 strict
  chains.

The final three conditions are essential. A dataset can require two documents
for answer production while still presenting no meaningful non-myopic root
choice because the setup document is directly named and trivially retrieved.

Failure closes this 2Wiki construction before any LLM call. Passing authorizes
only a separately preregistered, at-most-10-call development smoke. The 1,000
holdout rows stay sealed until both a development mechanism gate and a paired
policy design are frozen.

## Intended LLM-Native Test If Authorized

The LLM would generate natural-language support-chain hypotheses from the
question and visible document titles, then refresh those hypotheses after a
revealed paragraph. A non-myopic scorer would value the future support belief
induced by opening a setup paragraph. Paired controls would share generated
roots, observations, and call budgets:

- title/paragraph retrieval;
- myopic semantic root scoring;
- fixed-support depth-two scoring;
- model-aware depth-two scoring with LLM hypothesis refresh;
- seeded random root.

External endpoints would be exact ordered support-chain recovery and final
answer/supporting-fact accuracy. The LLM must remain load-bearing in semantic
hypothesis generation or transition scoring; a classical exact evidence graph
is an audit oracle only.

## Cost And Scheduling

- Opportunity audit: zero OpenRouter calls and zero cluster jobs.
- OatML remains paused until explicitly re-enabled.
- The live balance last verified after MuSiQue was `$66.123386116`.
- Preserve `$25` through Monday 2026-07-27. No paid 2Wiki stage is authorized
  by this document.
