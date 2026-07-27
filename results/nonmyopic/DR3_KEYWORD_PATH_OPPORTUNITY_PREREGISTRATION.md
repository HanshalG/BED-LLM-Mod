# DR3 Keyword-Path Opportunity Preregistration

## Status

Frozen before computing any retrieval outcome. This is a zero-call structural
gate for a source-grounded LLM-native sequential BED construction. A full pass
authorizes only a separately preregistered OpenRouter serving smoke and
development mechanics test. It is not itself a policy result.

## Source

- Official code: `NJU-LINK/DR3-Eval` at commit
  `86fed3760a8708d48121c4e9eaf0fddc939c6bef`.
- Official Hugging Face dataset revision:
  `4305f9129529d4510f485af6c997b69e1e85b88d`.
- English `query.jsonl` SHA-256:
  `52aa6a3ff3ca03f4da962d32a78785d3219aa921e2132d15196960a49afca203`.
- Length-prefixed combined SHA-256 of all 50 English 32k context JSON files:
  `2011d08d61530690ab7d4980612bd71ea81a9c31eae649df7a11124d6f845a82`.

The public release has 50 English tasks. Each task has a static 32k sandbox of
roughly 30--64 pages. Each page contains a construction `keyword`, a title, a
URL, and body text. Forty-nine tasks have exactly ten distinct nonempty
keywords; task `010` has nine. The benchmark's gold insight and useful-search
files are not released, so its native LLM-judge endpoint is not used.

## Quarantine And Split

Tasks `001`--`012` are mechanics-only because their prompts or sample pages
were displayed during source inspection. They can never enter opportunity,
development, holdout, or policy evidence.

The remaining task IDs were shuffled using Python `random.Random(24691)`:

- opportunity: 20 IDs, hash
  `c88bed3af2d6cfd5380b1f0907e5a6095304c586441ee942f596b74ef2843cb1`;
- development: 8 IDs, hash
  `3869d06cb649db66ec7c3182550ee374209cd96111698f7171cf78a3fdb08129`;
- holdout: 10 IDs, hash
  `37f955d04b37582d7acda03e8d235ce11779876b36744d9b0a036b302ebf8185`;
- complete shuffled-order hash:
  `ee884c9d690598bc525b3229855dc0b4984b441218a19aab27051dbfc82fb318`.

The opportunity audit may parse only the 20 opportunity prompts and context
files. Development and holdout bytes may be hash-verified but their content,
keywords, retrieval outcomes, and task records remain sealed.

## Exact Environment

The visible initial state is the released task query. A search action is a
free-text query. The deterministic observation is the top three BM25 pages,
showing only title and query-matching body snippets. The page's construction
keyword is hidden from query generation, retrieval, observations, and
continuation generation.

The exact utility of a history is the number of distinct hidden construction
keywords represented by its retrieved pages, divided by the task's number of
distinct keywords. This is a source-grounded research-facet coverage endpoint,
not a claim that every keyword is equally important to a final report.

The first search and second search each retrieve three pages. Second-step
retrieval excludes pages returned at the root.

## Target-Blind Search Tree

Each task receives at most 20 deterministic roots generated from:

- the full task query;
- its sentences and substantive clauses;
- its highest-IDF visible query terms; and
- two- to four-token windows containing those terms.

Each root observation produces at most 24 followups from high-IDF terms found
only in visible titles and query-matching body snippets. Candidate generation
never receives a page keyword, hidden label, development record, or holdout
record.

For every root, the audit exhausts this fixed followup bank. Ties use frozen
candidate order.

## Strict Opportunity

- Immediate value: distinct hidden keywords among the root's three pages.
- Pair value: distinct hidden keywords in the union of root and continuation
  pages.
- Greedy root: maximum immediate value, then maximum best pair value, then
  root order.
- Oracle two-step root: maximum best pair value, then maximum immediate value,
  then root order.

A strict non-myopic opportunity requires:

1. the roots differ;
2. the oracle two-step root has strictly lower immediate keyword coverage;
3. its best pair covers strictly more keywords than the greedy root and the
   greedy root's own best continuation; and
4. its continuation adds at least one new keyword.

## Frozen Gates

All conditions must pass:

- all 20 opportunity tasks complete;
- every task has at least 25 pages, 9 keywords, and 5 roots;
- at least 15 tasks have at least three distinct root top-1 pages;
- at least 10 tasks improve best immediate keyword coverage at depth two;
- mean oracle pair coverage is at least `.30`;
- mean pair-coverage gain over best immediate coverage is at least `.08`;
- at least 4 tasks meet the strict opportunity definition;
- strict tasks have a total non-myopic gap of at least 4 keywords; and
- mean normalized gap among strict tasks is at least `.10`.

No split, corpus scale, tokenizer, root width, retrieval width, followup width,
observation length, endpoint, tie break, or threshold changes after outcomes.
Failure closes this exact construction before model use.

## Conditional LLM-Native Study

Only a full structural pass may authorize a fresh protocol. The intended
development comparison is:

- a nonreasoning LLM generates semantic information-need hypotheses and root
  searches;
- each root observation causes the LLM to regenerate a path-dependent support
  and continuation search;
- a full-tree semantic scorer selects a root;
- an isolated myopic scorer sees only root observations;
- fixed-support, lexical, and seeded-random controls share the same retrieval
  budget; and
- hidden keyword coverage is the exact paired endpoint.

The serving smoke must precede science, use OpenRouter only, and cost at most
`$0.20`. A development mechanics test must be separately frozen and capped at
`$0.80`. Reasoning remains a naive baseline, not part of the BED policy.
