# BIRD Intent-World Manifest Result

Date: 2026-07-26. Outcome: **zero-call gate failed**.

## Frozen Gate

Before selecting any task, the BIRD-derived construction required at least 60
uninspected Mini-Interact tasks satisfying all of:

- three or four critical ambiguity points;
- at least two masked critical ambiguities;
- at least one knowledge/schema grounding ambiguity;
- at least one semantic/intent/lexical/syntactic ambiguity;
- at least one knowledge ambiguity;
- unique, nonempty terms and nonempty SQL snippets;
- no released solution, test, or follow-up endpoint.

The three task records already inspected semantically (`alien_1` through
`alien_3`) were excluded. Seed `24414` prospectively assigned 3 mechanics, 24
opportunity, 12 development, and the remainder to holdout. The gate also
required all four splits to be nonempty and the three mechanics tasks to span
three databases.

## Result

Only **39** tasks were eligible, below the frozen minimum of 60. The requested
3/24/12 allocation exhausted all 39 and left an empty holdout. The source hash
and three-database mechanics check passed, but the eligible-count and nonempty
split gates failed.

No threshold, eligibility rule, or split size was changed after observing this
count. No task query, ambiguity content, SQL content, or endpoint field was
emitted by the manifest. The exact construction is closed rather than amended
to fit the available data.

- OpenRouter calls: `0`
- OpenRouter spend: `$0`
- Database downloads: `0`
- OatML jobs: `0`

The reproducible failure artifact is
`bird_interact_intent_world_manifest/GATE_FAILURE.json`.
