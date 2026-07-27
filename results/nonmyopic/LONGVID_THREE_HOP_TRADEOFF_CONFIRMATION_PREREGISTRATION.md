# LongVidSearch Three-Hop Tradeoff Confirmation Preregistration

## Status

Frozen before loading caption text or computing retrieval outcomes for any of
the 40 three-hop reserve videos. This is a prospective confirmation of a
development-derived semantic root tradeoff, not a reinterpretation of the
failed source-order gate.

No model call is authorized unless every confirmation gate passes.

## Development Motivation

The frozen three-hop opportunity audit failed its required released-order
`0→1→2` mechanism: only 5/40 tasks contained that trajectory and none met the
strict definition.

A broader diagnostic was nevertheless positive on the already-open
opportunity block:

- 6/40 greedy and oracle roots differed;
- all six oracle roots had lower direct-answer support;
- all six achieved one additional necessary clip after three searches;
- the total final-coverage gap was six clips; and
- mean direct-answer sacrifice was `0.2041`.

This event was not preregistered and is not a result. It motivates exactly one
prospective confirmation on caption-unopened videos.

## Frozen Confirmation Set

Use the 40 `reserve` row IDs from
`LONGVID_THREE_HOP_OPPORTUNITY_PREREGISTRATION.md`, in their frozen order:

`867, 1168, 728, 925, 794, 1214, 2407, 2673, 2632, 2321, 1931, 262, 2807,
805, 444, 2740, 1848, 1139, 1997, 309, 382, 1086, 532, 557, 1949, 578, 707,
1030, 660, 2121, 2393, 2190, 1581, 2796, 2854, 2396, 1273, 1019, 1447,
631`.

The row hash is
`0b95e40d40d55c78c5e7b7ef7c8a540a536fc83e50b9d71ff23e18e554ce8937`.
The ordered reserve-video hash is
`49973b0c7c1e1d480e2035229f6ff326aae88f5d3eabba7dbbf70e35bde0388a`.

These videos are disjoint from the 40 opened two-hop videos, 40 opened
three-hop opportunity videos, 20 unopened three-hop development videos, and
22 caption-only fresh videos.

Only reserve captions may be returned through the frozen PyArrow `vid`
predicate. Development and fresh captions remain unopened.

## Unchanged Search And Values

Reuse the exact committed three-hop implementation:

- at most 20 question-derived BM25 roots;
- top-one retrieval;
- at most 12 observation-derived first continuations;
- at most 12 observation-derived second continuations;
- exclusion of previously retrieved clips;
- complete `20 x 12 x 12` tree;
- source-order candidate ties;
- greedy root by direct-answer coverage, gold indicator, final coverage, then
  root order; and
- oracle root by final coverage, direct-answer coverage, gold indicator, then
  root order.

No query grammar, tokenizer, width, tie break, answer proxy, or evidence
endpoint changes are permitted.

## Semantic Tradeoff

A task is a strict semantic tradeoff exactly when:

1. greedy and oracle root indices differ;
2. oracle direct-answer coverage is strictly lower than greedy direct-answer
   coverage; and
3. oracle final necessary-clip count is strictly higher than the greedy root's
   own best three-search count.

Because a final count above the greedy count must come from an enumerated
three-search trajectory, both continuation queries remain
observation-conditioned by construction. No released evidence-order condition
is imposed.

For strict tasks:

- gap is oracle final clip count minus greedy final clip count;
- sacrifice is greedy direct-answer coverage minus oracle direct-answer
  coverage.

## Frozen Gates

All conditions must pass:

- all 40 tasks complete with at least 60 captions, 5 roots, and 2 answer terms;
- at least 30 tasks have at least 3 distinct root top-one clips;
- at least 15 tasks gain at least one necessary clip by depth three;
- mean oracle three-clip coverage is at least `.40`;
- mean coverage gain over best immediate gold coverage is at least `.20`;
- at least 5/40 tasks are strict semantic tradeoffs;
- strict total gap is at least 5 clips; and
- mean strict direct-answer sacrifice is at least `.15`.

The `5/40`, five-clip, and `.15` thresholds are frozen from the development
diagnostic before reserve-caption access. Results are reported for all 40
reserve tasks with no subset, category selection, or threshold repair.

Failure closes this exact semantic-tradeoff route. Passing establishes only a
classical structural opportunity and authorizes a separately frozen 10-call
nonreasoning OpenRouter serving smoke capped at `$0.20`.

## Budget

The confirmation uses zero API calls and zero spend. OpenRouter remains the
only permitted model route; OatML/Slurm is forbidden. At least `$25` remains
protected through Monday under the authenticated `$33.574042594` balance and
the `$8.50` new-spend ceiling.

