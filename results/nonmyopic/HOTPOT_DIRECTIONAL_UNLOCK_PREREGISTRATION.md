# HotpotQA Directional Unlock Audit Preregistration

Date frozen: 2026-07-25

## Purpose

Test whether HotpotQA distractor bridge questions contain an externally
annotated, path-order mechanism suitable for LLM-native non-myopic BED.

The proposed environment reveals one hidden paragraph per action. An enabling
support paragraph can name the answer-bearing support document and thereby
improve the next semantic belief/retrieval state. Inspecting the answer-bearing
paragraph first may provide immediate answer evidence without exposing the
enabling document. This audit tests only whether that asymmetric structure
exists. It makes no LLM or policy claim.

## Source And Frozen Split

- Official dataset: HotpotQA distractor validation split.
- Official project: `https://hotpotqa.github.io/`.
- Hugging Face auto-converted mirror:
  `hotpotqa/hotpot_qa`, configuration `distractor`, split `validation`.
- Parquet SHA-256:
  `c20b638ca82b21d04fe12e14ff417ad05153d4d215a65de54497fca4e972f7c6`.
- Rows: 7,405 total; 5,918 have metadata type `bridge`.
- Selection seed: `24350`.

Before any answer, supporting-fact, question, or context column was read, the
sorted bridge IDs were shuffled with Python `random.Random(24350)` and split:

- opportunity: first 500 IDs, ordered-list SHA-256
  `e324df5b63a8cad523801ffa64111cd377166e0875a5a328599173f9ae25e653`;
- development: next 100 IDs, SHA-256
  `72680f9764936f9595ab413b9055fdcd20c4b6ff77770ddf51fddcf4562c873c`;
- untouched holdout: remaining 5,318 IDs, SHA-256
  `e05826f43243d10e2c15266770425c640282c73c1d224c4d18081650c6f6f2da`.

This audit may read endpoints only for the 500 opportunity IDs. Development and
holdout rows remain sealed.

## Frozen Direction Rule

Normalize text by Unicode NFKD, ASCII case-folding, replacing underscores with
spaces, retaining alphanumeric tokens, and collapsing whitespace.

For a context title, use two aliases:

1. the complete normalized title;
2. the normalized prefix before a final parenthetical disambiguator.

Ignore aliases shorter than three characters or containing no alphanumeric
token. A paragraph mentions a candidate title when a title alias appears as a
contiguous token sequence in its normalized full text.

For each opportunity row:

1. Require `type == bridge`.
2. Require exactly two distinct supporting-fact titles, each occurring exactly
   once among the ten context titles.
3. Exclude answers `yes` and `no`, normalized answers shorter than three
   characters, and rows where the answer phrase occurs in supporting sentences
   from zero or both support documents.
4. The support document whose annotated supporting sentences contain the answer
   phrase is the **answer document**. The other is the **enabling document**.
5. Compute all other context titles mentioned by each support paragraph.
6. A strict directional unlock exists only when the enabling paragraph mentions
   exactly one other candidate title and it is the answer document, while the
   answer paragraph does not mention the enabling title.

The transition policy is target-independent: after inspecting a paragraph,
follow its sole mentioned candidate title when exactly one exists; otherwise
there is no deterministic follow-up. Under a strict unlock, enabling-first
retrieves both annotated supports in two actions, whereas answer-first does not.

Also record a nonsemantic title baseline: BM25 over the ten candidate titles
with the question as query, original context order for ties. It receives no
paragraph text or endpoint.

## Gates

The zero-call opportunity audit passes only if all hold:

- source hash, row count, bridge count, and all three split hashes reproduce;
- all 500 opportunity rows are processed with no development or holdout access;
- at least 40/500 rows meet the strict directional-unlock definition;
- at least 30 strict unlocks are medium or hard difficulty;
- the title-only BM25 root is not the enabling document on at least 30 strict
  unlocks;
- at least 20 strict unlocks have neither support title stated verbatim in the
  question;
- enabling-first minus answer-first support-document coverage is exactly one
  on every strict unlock.

Failure closes this HotpotQA construction before any LLM call. Passing only
authorizes a separate preregistered 10-call serving/mechanism smoke on the 100
development IDs. The 5,318-row holdout cannot be accessed until that mechanism
and a powered paired policy gate pass.

## Intended LLM-Native Test If Authorized

The LLM would generate natural-language answer/support-chain hypotheses from
the question and visible titles, refresh them after a revealed paragraph, and
score the value of the induced future belief state. Controls would share root
candidates and paragraph observations:

- myopic semantic relevance;
- fixed-support depth two;
- model-aware depth two with regenerated hypotheses;
- random root with the same receding continuation rule.

Exact supporting-document coverage and answer/supporting-fact F1 would be
external endpoints. The present audit does not authorize those calls.

## Cost And Scheduling

- Opportunity audit: zero OpenRouter calls and zero cluster jobs.
- OatML remains paused.
- Verified pre-audit live balance: `$46.369749126`; `$25` remains protected
  through Monday.
