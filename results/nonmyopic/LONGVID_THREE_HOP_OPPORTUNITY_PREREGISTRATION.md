# LongVidSearch Three-Hop Opportunity Preregistration

## Status

Frozen before opening caption text for any selected three-hop video or computing
any three-hop retrieval outcome. This is a new source-defined task family,
motivated by the frozen two-hop result rather than a repair of it.

No model call is authorized unless every structural gate passes.

## Motivation

The two-hop audit found strong sequential retrieval but only one strict root
tradeoff. Its mechanism was exact: the greedy answer-side root and its one
remaining search recovered both necessary clips on 23/40 tasks.

LongVidSearch separately releases 718 three-hop tasks whose three distinct
clips all passed the benchmark's necessity check. With exactly three searches,
an answer-side root must backtrack through two missing clips, while a
bridge-first root can follow the ordered evidence chain. This prospectively
tests whether additional source-native chain depth creates a genuine first
action conflict.

## Source And Freshness

Source files, hashes, caption semantics, and the 22 caption-only fresh-video
reserve are exactly those in
`LONGVID_BRIDGE_PATH_OPPORTUNITY_PREREGISTRATION.md`.

All official QA records are development-only because the monolithic QA JSON was
materialized during source inspection. Caption text was returned only for the
40 two-hop opportunity videos. The new three-hop split excludes those videos
and the 22 caption-only fresh videos.

Caption loading again uses a PyArrow `vid` predicate before conversion to
Python. Only the 40 three-hop opportunity videos may be returned. Three-hop
development/reserve videos and the caption-only fresh reserve remain unopened.

## Video-Disjoint Split

Rows `0..99` are mechanics-only. Eligible records are released `3-Hop`
`Causal_Inference` or `State_Mutation` tasks whose videos were absent from the
two-hop opportunity set.

For each category, preserve released row order, retain the first eligible row
per video, and shuffle sorted video IDs:

- Causal-Inference seed `270733`, 167 available videos, shuffled-order hash
  `dbd097b24ab80762da62683457c493b17e095fc40515a653ccd75b501483ab4e`;
- State-Mutation seed `270734`, 70 videos remaining after prior selections,
  shuffled-order hash
  `d8d80766a3225fcb3c0889eea36465603b973c62a24c98dc131d5efe359829cf`.

For each category, assign the first 20/next 10/next 20 videos to
opportunity/development/reserve. The resulting row hashes are:

- opportunity 40:
  `1e7ab608a86bf6281a88779c633fc5cdcfc208065fe263ef78970bc3704a142c`;
- development 20:
  `e3606098b77a05faabb13d703204eb5de8c0d05352d948708655f5db51cea1a3`;
- reserve 40:
  `0b95e40d40d55c78c5e7b7ef7c8a540a536fc83e50b9d71ff23e18e554ce8937`.

The opportunity video hash is
`55d038286dad0eafefd6b11fd66a41ca549617639bf420ac8b2a375e61218f30`.
All 100 videos are distinct and none overlaps the two-hop opportunity videos.

## Exact Three-Search Tree

The initial question, BM25 caption corpus, top-one retrieval, source-order tie
break, hidden fields, and direct-answer diagnostic match the frozen two-hop
protocol.

Each task receives:

- at most 20 question-derived roots;
- at most 12 first continuations generated from the root caption; and
- at most 12 second continuations generated from the newly retrieved second
  caption.

Every continuation includes a term visible in its immediately preceding
observation and absent from the initial question and preceding query. Retrieved
clips are excluded from later searches. The audit exhausts the complete
`20 x 12 x 12` tree with frozen candidate-order ties.

The source evidence list is treated as ordered:

- position 0: initial bridge;
- position 1: intermediate bridge;
- position 2: answer-side evidence.

## Values And Strict Opportunity

- Greedy root: maximum direct-answer token coverage, then gold-root indicator,
  then best three-search evidence coverage, then root order.
- Oracle root: maximum best three-search evidence coverage, then direct-answer
  coverage, then gold-root indicator, then root order.
- Final coverage: distinct gold clips in the three retrieved clips, divided by
  three.

A strict opportunity requires:

1. greedy and oracle roots differ;
2. the oracle root has strictly lower direct-answer coverage;
3. the greedy root retrieves evidence position 2 first;
4. the oracle trajectory retrieves positions `0`, `1`, and `2` in order;
5. the oracle covers all three clips while the greedy root's own best
   continuation tree does not; and
6. both oracle continuations contain valid observation-derived terms.

## Frozen Gates

All conditions must pass:

- all 40 tasks complete with 60 captions, 5 roots, and 2 answer tokens;
- at least 30 tasks have at least 3 distinct root top-1 clips;
- at least 15 tasks gain at least one gold clip by depth three;
- at least 10 tasks recover the complete ordered `0→1→2` chain;
- mean oracle three-clip coverage is at least `.40`;
- mean coverage gain over best immediate gold coverage is at least `.20`;
- at least 5 tasks meet the strict opportunity definition;
- strict tasks have a total oracle-over-greedy gap of at least 5 clips; and
- mean direct-answer sacrifice among strict tasks is at least `.15`.

No task subset, evidence order, search width, candidate grammar, tokenizer,
proxy, tie break, or threshold changes after outcomes. Failure closes the exact
three-hop construction before model use.

## Conditional Paid Route

A full pass authorizes only a separately frozen 10-call nonreasoning OpenRouter
serving smoke capped at `$0.20`. The intended LLM role remains semantic belief
generation, path-dependent support regeneration, likelihood assignment, and
full-tree scoring. Reasoning remains a naive baseline.

No OatML/Slurm work is permitted. At least `$25` of the authenticated
`$33.574042594` balance remains reserved through Monday; the new-spend ceiling
is `$8.50`.
