# LongVidSearch Four-Hop Tradeoff Confirmation V2 Preregistration

## Status

Frozen before loading caption text or computing retrieval outcomes for any of
the 40 four-hop confirmation videos.

V1 remains a formal failure. V2 changes only the prospective handling of
answers for which the direct-answer diagnostic is not statistically scorable.
No V1 row is dropped or reinterpreted as a V1 pass.

## V1 Motivation

The V1 opportunity block passed every scientific gate:

- 10/40 strict lower-immediate, higher-final roots;
- 11 clips of total strict gap;
- mean strict sacrifice `0.2650`;
- 39/40 depth-four gain;
- mean oracle coverage `0.6500`; and
- mean coverage gain `0.4125`.

It failed because one non-strict task had zero non-stopword answer terms,
making the direct-answer diagnostic undefined and completeness `39/40`.

## Frozen Confirmation Set

Use all 40 `confirmation` rows from the committed four-hop split:

`236, 1482, 2958, 1522, 218, 1314, 1404, 1415, 749, 900, 1422, 2703, 883,
1347, 1903, 1867, 2514, 485, 855, 1295, 2923, 453, 1207, 2989, 134, 1103,
2728, 549, 1817, 1630, 2433, 2642, 2345, 319, 120, 1476, 908, 561, 282,
2596`.

Frozen hashes:

- row IDs:
  `1cecd8a4318a0c283f44d18feff1551babba6eaff2218f508fc1141141944453`;
- ordered video IDs:
  `e0959f11ec5a9c5198b9fa468a3b9681e148623ac0cbab7018206f1bc5300d6e`.

The 22 four-hop reserve videos, 20 three-hop development videos, and 22
caption-only fresh videos remain caption-unopened.

## Unchanged Mechanics

Reuse the exact committed V1:

- root and continuation grammars;
- 20 roots and 8 continuations per stage;
- exact `20 x 8 x 8 x 8` search;
- BM25 top-one retrieval;
- exclusion of prior clips;
- source-order candidate ties;
- direct-answer tokenization;
- greedy and oracle definitions;
- necessary-clip endpoint; and
- strict lower-immediate, higher-final criterion.

No search, parser, tokenizer, answer term, root, coverage, tie, or strict-event
change is permitted.

## V2 Scorable Eligibility

Every task remains in the retrieval aggregates and must have at least 60
captions and 5 roots.

A task is answer-scorable exactly when it has at least two unique non-stopword
answer terms under the unchanged tokenizer. Only answer-scorable tasks may
count as strict tradeoffs or contribute to strict sacrifice/gap. Unscorable
tasks remain in diversity, depth-gain, oracle-coverage, and coverage-gain
aggregates.

At least 38/40 confirmation tasks must be answer-scorable. This threshold is
frozen before confirmation-caption access. V1 had 39/40 scorable tasks.

## Frozen Gates

All conditions must pass:

- all 40 tasks have at least 60 captions and 5 roots;
- at least 38/40 tasks are answer-scorable;
- at least 30 tasks have at least 3 distinct root top-one clips;
- at least 20 tasks gain at least one necessary clip by depth four;
- mean oracle four-clip coverage is at least `.45`;
- mean coverage gain is at least `.25`;
- at least 6/40 tasks are eligible strict tradeoffs;
- eligible strict total gap is at least 6 clips; and
- mean eligible strict direct-answer sacrifice is at least `.15`.

No pooling with V1, category subset, threshold repair, or confirmation rerun is
permitted.

A full pass establishes a replicated structural opportunity only. It
authorizes a separately committed exact 10-call nonreasoning OpenRouter
serving smoke capped at `$0.20`; it does not authorize a policy run.

## Budget

Confirmation calls: `0`. Cost: `$0`. OatML/Slurm is forbidden. At least `$25`
remains protected through Monday under the authenticated `$33.574042594`
balance and `$8.50` new-spend ceiling.

