# LongVidSearch Bridge-Path Opportunity Preregistration

## Status

Frozen before reading any caption text or computing any retrieval outcome. This
is a zero-call structural gate for an LLM-native sequential BED construction.
A full pass authorizes only a separately frozen serving smoke. Failure closes
this exact construction before any model call.

## Source

- Official code/data repository: `yrywill/LongVidSearch` at commit
  `4aa5620e06ca1bc5cc7cfce2b3e3ed0a5ae82d4e`.
- `full-QA(3000).json` SHA-256:
  `370711f1299202cd40d559440859476bf3a2d2ca74540dad5226786558ee123b`.
- `video-caption.parquet` SHA-256:
  `0f2ce94265b7050eaee5de239a760c1a0f0762c1754cbb3bd7b45d00774b6c68`.
- Sorted `shasum` manifest hash for the dataset tree:
  `9c617b29fdacd03daeb0902eddf6d06636b56f31dbdd6a02cc0769ea6705a32d`.
- Sorted `shasum` manifest hash for the released-embedding tree:
  `fc70494504b6dfaa8f4f129404d04cabfa2f7a778f47d7055845841630064df9`.

The release contains 3,000 unique QA records over 444 videos and 40,804
one-indexed caption slices over 467 videos. Every QA video and every listed
evidence slice has a released caption and embedding. Evidence lists contain
exactly 2, 3, or 4 unique clips and match the released hop label.

LongVidSearch's construction explicitly rejects pseudo-multihop questions and
runs a necessity check requiring every listed clip: removing any one must make
the full answer underdetermined. This audit treats those released evidence
labels as source-grounded but not human-perfect.

## Freshness Boundary

The monolithic QA JSON was materialized during structural source inspection, so
all 3,000 official QA records are development-only. No official QA task can
become confirmatory or headline evidence.

The caption Parquet was inspected only through its `vid` and `slice_num`
columns. It contains 23 videos with no released QA record; 22 also have released
retrieval embeddings. Their caption text has not been returned to Python,
printed, prompted, scored, or persisted:

- caption-only video-ID hash:
  `bbb921ad8caa948a41b688fe166af49813aa8246cfe18749ba28ececf709bfdd`;
- embedding-eligible 22-video hash:
  `eb964ad3fa8c2f34eba823467d535490ae62fe01a91f2a8c23a1011915c45855`.

The audit reads captions with a PyArrow `vid` predicate before converting rows
to Python. Only opportunity video IDs may be returned. The 22 fresh videos are
excluded from audit records and remain reserved for a separately frozen
fresh-task construction and confirmation if development mechanics pass.

## Development Opportunity Set

Rows `0..99` are permanently mechanics-only. Eligible rows are released
`2-Hop` tasks in `Causal_Inference` or `State_Mutation`.

Selection is video-disjoint:

1. preserve released row order and choose the first eligible row per video and
   category;
2. shuffle Causal-Inference videos with seed `270731`;
3. assign the first 20/next 10/next 20 to
   opportunity/development/reserve;
4. remove those 50 videos from the State-Mutation pool;
5. shuffle remaining State-Mutation videos with seed `270732`; and
6. assign another 20/10/20 rows.

The 40 opportunity row indices have hash
`092b0337cad5d0fce436e1378640bd1e8580a2fe29953a25e189593ba17bbbb6`
and video hash
`57c0c3a44eb17ac32e3d4fe8c4512881875196d6628a672cd202c111370314e1`.
The 20 development indices have hash
`1c886e1b58c19e6774b7d8874af3feca6db587325f3f9884f46b00fe8fd70dca`;
the 40 reserve indices have hash
`535ff26012218733c88fadbd0677cf9ab42051a5ef8e9c857b9e4ad670a09a4d`.
All 100 selected records use distinct videos.

Only the opportunity records and captions may be analyzed by this gate.
Development and reserve are already development-only but remain unopened by
the audit to reduce adaptive reuse.

## Exact Two-Search Environment

Each video is a separate corpus of 60--100 released clip captions. A free-text
search action returns the top one BM25 caption with deterministic source-order
tie breaking. The second search excludes the first clip.

The visible initial state is only the released question. The first observation
is the returned caption, truncated to 2,000 characters. The answer, evidence
slice IDs, reasoning chain, and all unretrieved captions are hidden from
candidate generation.

The source evidence list is treated as ordered: element zero is the bridge hop
and element one is the answer-side hop. This follows the released generation
format, which emits evidence IDs alongside an ordered reasoning chain.

## Target-Blind Search Tree

Each task receives at most 20 deterministic roots generated from:

- the full question;
- its sentences and substantive clauses;
- its highest-IDF visible question terms; and
- two- to four-token question windows containing those terms.

Each root observation produces at most 24 followups from high-IDF terms visible
only in the retrieved caption. Every followup must contain at least one term
absent from the initial question and root query. The audit exhausts the frozen
tree; ties use candidate order.

## Immediate And Final Values

For structural gating only, direct answer evidence is the fraction of unique,
non-stopword reference-answer tokens present in the root caption, set to zero
when the root clip is not gold evidence. Candidate generation never receives
the answer or this score.

- Greedy root: maximum direct-answer evidence, then gold-clip indicator, then
  best pair coverage, then root order.
- Oracle root: maximum two-search gold-evidence coverage, then direct-answer
  evidence, then gold-clip indicator, then root order.
- Pair coverage: number of distinct listed evidence clips retrieved in the two
  searches, divided by two.

The direct-answer score is an opportunity diagnostic, not the later BED policy
score. Any paid policy must use target-blind LLM-generated hypotheses and
likelihoods.

## Strict Bridge Opportunity

A task is strict only if:

1. greedy and oracle roots differ;
2. the oracle root has strictly less direct-answer evidence;
3. the greedy root retrieves the answer-side evidence clip first;
4. the oracle root retrieves the bridge evidence clip first;
5. the oracle's observation-conditioned followup retrieves the answer-side
   evidence clip;
6. the oracle pair covers both evidence clips while the greedy root and its own
   best continuation do not; and
7. the oracle continuation contains a visible observation term absent from the
   initial and root queries.

## Frozen Gates

All conditions must pass:

- all 40 opportunity tasks complete with at least 5 roots and 2 answer tokens;
- at least 30 tasks have at least 3 distinct root top-1 clips;
- at least 15 tasks gain a gold clip at depth two;
- at least 10 tasks recover the ordered bridge-to-answer chain;
- mean oracle pair coverage is at least `.40`;
- mean pair-coverage gain over best immediate gold coverage is at least `.15`;
- at least 5 tasks meet the strict bridge definition;
- strict tasks have a total oracle-over-greedy gap of at least 5 clips; and
- mean direct-answer sacrifice among strict tasks is at least `.15`.

No task subset, root width, retrieval width, followup width, tokenizer,
immediate proxy, evidence order, tie break, or threshold changes after outcomes.

## Conditional LLM-Native Route

A full gate pass authorizes a 10-call OpenRouter serving smoke capped at `$0.20`.
The intended policy is nonreasoning:

- the LLM generates possible answers and missing semantic evidence needs;
- each candidate root caption causes the LLM to regenerate that support and a
  path-dependent continuation;
- a full-tree scorer values the quality of the induced next belief state;
- a myopic scorer, compute-matched fixed-support scorer, and seeded random
  policy share exactly the same retrieval budget; and
- evidence recall and answer support are exact paired endpoints.

Reasoning is reserved for a naive baseline. No OatML/Slurm work is permitted.
The authenticated OpenRouter balance is `$33.574042594`; at least `$25` remains
reserved through Monday and new pre-Monday spend is capped at `$8.50`.

If development mechanics pass, fresh confirmation must be separately frozen
before opening any of the 22 caption-only videos. Task generation, selection,
necessity validation, policy, controls, endpoint, and spend must all be fixed
before those captions enter model prompts.
