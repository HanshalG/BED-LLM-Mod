# LongVidSearch Four-Hop Tradeoff Opportunity Preregistration

## Status

Frozen before loading caption text or computing retrieval outcomes for any
selected four-hop video. This tests whether one additional source-native
necessary clip makes semantic first-action conflicts prevalent enough for a
non-myopic LLM policy experiment.

No model call is authorized by this document.

## Source And Exclusions

Use the official LongVidSearch files and hashes already frozen in the two-hop
and three-hop protocols.

Exclude:

- QA rows `0..99`;
- every video in the frozen two-hop opportunity/development/reserve split;
- every video in the frozen three-hop
  opportunity/development/reserve split; and
- all 22 caption-only fresh videos.

Eligible records are official `4-Hop` tasks in `State_Mutation`,
`Causal_Inference`, or `Global_Summary`. Retain the first released row per
video within each category.

All official QA records are development-only. Caption text for all selected
four-hop videos is unopened at freeze time.

## Video-Disjoint Split

Process categories sequentially so a video is assigned once:

1. `State_Mutation`: seed `270735`, 29 videos, shuffled-order hash
   `c09872a82f08169463343713b9db4816bd6d085d8f35423230c9bb7fe7a484c8`;
2. `Causal_Inference`: seed `270736`, 43 videos after state-video exclusion,
   shuffled-order hash
   `f433e0b65f75fe64af406d7228046f87146ae79b4a0233110638fd3f52abe08e`;
3. `Global_Summary`: seed `270737`, 30 videos after prior-category exclusion,
   shuffled-order hash
   `4a6861a74efc524b4fe3646a67c03b1e8e096b8ee1e118dcb6ada213adcc911a`.

Allocate per category:

| Category | Opportunity | Confirmation | Reserve |
|---|---:|---:|---:|
| State Mutation | 14 | 14 | 1 |
| Causal Inference | 16 | 16 | 11 |
| Global Summary | 10 | 10 | 10 |

Frozen hashes:

| Split | Rows | Row hash | Ordered video hash |
|---|---:|---|---|
| Opportunity | 40 | `6c27da94e920e4c7c98df22e549612bef9f6f31e2879f4a30f53e9b7d5bf60b2` | `0ac0cd4bba6f626b12446d69d82e40d34d7651a86caf2eedbfca28be5bdef59a` |
| Confirmation | 40 | `1cecd8a4318a0c283f44d18feff1551babba6eaff2218f508fc1141141944453` | `e0959f11ec5a9c5198b9fa468a3b9681e148623ac0cbab7018206f1bc5300d6e` |
| Reserve | 22 | `fb7ea1295874f273bc8d8ce163630462bd98533b557a30cb946629e546ca9c0c` | `35c1a2747b5e4376c3e2c3dc067a2eec08dd29b4bc9b8421cdebc24a2d17ca7f` |

Only the 40 opportunity captions may be returned through the PyArrow `vid`
predicate. Confirmation and reserve captions remain unopened.

## Exact Four-Search Tree

For each task:

- construct at most 20 roots with the committed question-only grammar;
- retrieve one caption for the root;
- construct at most 8 first continuations from that caption;
- construct at most 8 second continuations from the second caption;
- construct at most 8 third continuations from the third caption; and
- retrieve one new caption after each continuation, excluding all previously
  retrieved slice IDs.

Each continuation must contain a term visible in the immediately preceding
caption and absent from the initial question and preceding query. Exhaust the
complete `20 x 8 x 8 x 8` tree. Candidate order breaks all ties. No query,
caption, answer, or evidence text is written to the public artifact.

The four released evidence slices are treated as a necessary set, not as a
prerequisite order.

## Greedy, Oracle, And Strict Tradeoff

- Greedy root: maximum direct-answer token coverage in the first retrieved
  gold caption, then first-caption gold indicator, then best four-search
  necessary-clip count, then root order.
- Oracle root: maximum best four-search necessary-clip count, then
  direct-answer coverage, first-caption gold indicator, then root order.

A strict tradeoff requires:

1. greedy and oracle roots differ;
2. oracle direct-answer coverage is strictly lower; and
3. oracle best final necessary-clip count is strictly higher.

Gap and direct-answer sacrifice use the same definitions as the frozen
three-hop confirmation.

## Frozen Gates

All conditions must pass:

- all 40 tasks complete with at least 60 captions, 5 roots, and 2 answer terms;
- at least 30 tasks have at least 3 distinct root top-one clips;
- at least 20 tasks gain at least one necessary clip by depth four;
- mean oracle four-clip coverage is at least `.45`;
- mean coverage gain over best immediate gold coverage is at least `.25`;
- at least 6/40 tasks are strict tradeoffs;
- strict total gap is at least 6 clips; and
- mean strict direct-answer sacrifice is at least `.15`.

The opportunity block is development-only. Failure closes the exact four-hop
construction. A full pass authorizes only a separately committed zero-call
confirmation using all 40 frozen confirmation videos under the same
implementation and thresholds. No paid LLM stage is authorized until that
confirmation also passes.

## Budget

Calls: `0`. Cost: `$0`. OatML/Slurm is forbidden. The authenticated balance,
`$25` Monday reserve, and `$8.50` new-spend ceiling are unchanged.

