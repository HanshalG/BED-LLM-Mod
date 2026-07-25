# SWE-QA-Pro Non-Myopic Code-Search Opportunity Audit

## Status

Frozen before inspecting any question or reference-answer text and before
computing any retrieval outcome. This is a zero-call gate for a real,
LLM-native sequential BED environment. Failure closes this construction before
model use; passing authorizes only a separately preregistered, sub-`$0.50`
development smoke.

## Source And Split

- Official code: `TIGER-AI-Lab/SWE-QA-Pro` at commit
  `93ac6a4f3af3fe3f86580f62142f47e97e2cc897`.
- Official benchmark: `TIGER-Lab/SWE-QA-Pro-Bench` at revision
  `596892dac60b6f500f01a7dc2becb9f66593b7b7`.
- `data/test.jsonl` SHA-256:
  `bba4aade95e707d012e11e622a40687bc0efd8cc509ed4a7dd3ba24c6e365737`.
- Exactly 260 rows and 26 repositories, each checked out at its released
  benchmark commit.
- Content-blind Python shuffle seed: `24364`.
- Opportunity: 120 row indices, hash
  `ed46d42df3fd0fd2e208d477503c82e572673b158978a2170bbcd67952099b05`.
- Development: 40 row indices, hash
  `aaf9a5e1940298641cdb0ba21921cca59f47120f6189e50f5887ccad80f47663`.
- Holdout: 100 row indices, hash
  `8b30c97453fc600e9ed0f528bbf64bf1d004151a2e45963c86a768648743c608`.
- Full shuffled-order hash:
  `bf8b3a3c873f181cea4023ecb19257310b56c95522a866ce8a59a7c6b2d404ef`.

The shuffle and hashes were fixed from row numbers only. Repository names,
commit IDs, cluster IDs, and QA-type labels were used only to verify metadata.
The opportunity script skips development and holdout lines without parsing
them. Their question and answer text are not eligible for this audit.

## External State And Endpoint

The external state is the exact source tree at the released commit. The
visible initial state is the benchmark question. A search observation is the
top three repository files under deterministic BM25 plus query-matching source
windows from those files. The second search excludes first-step files.

The hidden endpoint is the set of repository files explicitly cited by the
released reference answer:

- an exact repository-relative path occurring in the answer; or
- a filename occurring in the answer only when that basename is unique in the
  repository, has a stem of at least five characters, and is not a generic
  filename such as `README.md`, `setup.py`, or `__init__.py`.

Tasks with fewer than two recovered evidence files are unusable. No answer
text, answer token, evidence path, or QA label enters query generation,
retrieval, or continuation generation.

## Search Tree

The repository corpus includes tracked text/code files from a frozen suffix
allowlist, excludes build/vendor/virtual-environment paths, caps files at 512
KB and indexed source at 120,000 characters, and repeats path tokens six times.

Each question produces at most 24 deterministic root queries from:

- the full question and its sentences;
- backtick and quoted spans;
- high-IDF question tokens; and
- two-, three-, and four-token windows containing a high-IDF question token.

BM25 returns three files per root. The visible observation consists of the
path and up to 45 source lines in radius-two windows around the strongest
query-matching lines. At most 32 followups are generated from high-IDF terms
found only in that observation, using per-file and aggregate expansions.
Each followup retrieves three previously unseen files.

## Values And Strict Opportunity

Utility is reference-evidence file coverage.

- Immediate value: number of evidence files among a root's three files.
- Pair value: evidence files in the union of root and followup results.
- Greedy root: maximum immediate value, then maximum best pair value, then
  frozen root order.
- Non-myopic oracle root: maximum best pair value, then maximum immediate
  value, then frozen root order.

A strict opportunity requires all of:

1. the roots differ;
2. the non-myopic root has strictly lower immediate evidence coverage;
3. its best pair covers strictly more evidence than the greedy root plus the
   greedy root's own best continuation; and
4. its continuation adds at least one new evidence file.

This tie-breaking is conservative: a pair-optimal root with equal or better
immediate value prevents the task from counting as a forced non-myopic
tradeoff.

## Frozen Gates

All gates must pass:

- all 120 opportunity rows complete;
- at least 72 tasks expose at least two evidence files and three roots;
- every usable task has at least three roots;
- immediate search fully saturates at most 70% of usable tasks;
- at least 18 usable tasks gain evidence at depth two;
- at least 12 tasks meet the strict opportunity definition;
- at least 8 strict tasks expose at least three evidence files;
- strict tasks span at least 8 repositories; and
- mean normalized non-myopic gap on strict tasks is at least `.15`.

No threshold, split, tokenizer, corpus filter, retrieval width, continuation
count, path weight, evidence parser, or tie-breaking rule changes after
outcomes. A failed gate permits a result write-up only, not a development or
holdout LLM run.

## Conditional LLM-Native Smoke

Only a full gate pass may authorize a fresh preregistration on development
rows. The intended policy comparison is compute-matched:

- myopic LLM: scores each generated root from the question and immediate
  predicted evidence/belief value;
- non-myopic LLM: predicts the code hypotheses and likely identifier unlocks
  induced by each root, then scores the best observation-conditioned
  continuation;
- random-root control with the same continuation budget.

The LLM must generate semantic code hypotheses and continuation value; exact
retrieval and released evidence remain external. No reasoning trace is
required for the environment. Any smoke is capped below `$0.50`, with the
project's protected `$25` OpenRouter reserve preserved through Monday.
OatML is not used.
