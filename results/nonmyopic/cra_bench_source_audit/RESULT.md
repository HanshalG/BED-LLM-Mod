# CRA-Bench Source Audit

## Result

CRA-Bench is promising for conversational recommendation, but the current
release does not support a defensible paid non-myopic BED experiment.

The hash-pinned release contains 750 task rows: 250 underlying user/target
worlds repeated at easy, medium, and hard difficulty. The hidden user profile,
visible recommender profile, and exact target are constant across each set of
three variants. The hard split gives every task a two-turn patience budget and
contains 244 distinct target products across five domains.

The release does not contain the user-simulator implementation or prompt,
retrieval runner, product catalog, catalog reconstruction/filtering script,
reference policy, or a linked paper. Exact target products and metadata are
present only under evaluation-only fields.

## Decision

Do not turn the 244 evaluation targets into the policy's closed hypothesis
bank. That would leak the benchmark target pool. Do not invent response maps or
claim compatibility with the unreleased simulator.

The route is closed until an official runner and catalog become available.
At that point, the hard split's two-turn patience budget makes a target-blind
greedy-gap audit worthwhile before any model calls.

No OpenRouter calls were made and no task content, target IDs, or target
metadata is present in the public audit artifact.

Public analysis SHA-256:
`4cfee5aff6e3973c37b4aa63868cf22d501361a5138fa77d9be555f2ba5f0144`.
