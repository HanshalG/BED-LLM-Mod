# tau-Knowledge Paired Belief-Bottleneck Smoke

## Status

Frozen before any belief-bottleneck response. This is a ten-call causal and
efficacy gate on the two public V3 smoke trees. Failure closes the construction
before old-tree development or policy evaluation.

## Motivation

The prior refreshed-belief shuffle did not hurt the full semantic scorer. That
scorer also saw the customer opening, realized query, first results, followup
queries, and document evidence, so it could bypass the generated belief state.

This gate removes that bypass. The scorer sees only one generated
information-need state and four candidate returned-evidence sets. Customer
opening, initial beliefs, all query text, first results, required-document
labels, and endpoints are hidden.

## Frozen Data and Intervention

- Source: the exact two-task V3 smoke artifact, SHA-256
  `61c466ead49a5545abd54dd28e1a2fd7ad070287f5108684a3b2643befc40a01`.
- Ten rows: five realized roots per task.
- Aligned state: that root's eight generated refreshed information needs.
- Shuffled state: the next root's refreshed needs in cyclic root order within
  the same task.
- The aligned and shuffled states share the exact same four candidate evidence
  sets.
- Seed `24348` assigns aligned state to blinded label A on exactly five rows
  and label B on exactly five rows.

Every candidate exposes only its three document titles and first 500
characters of each excerpt. Document IDs, BM25 scores, candidate query text,
first-result evidence, and endpoint labels are omitted.

## Paired Scorer

GPT-5.4, temperature zero, explicit non-reasoning, returns eight
count-dominant scores in one flat JSON object: four followups under state A and
four under state B. A document scores only when its supplied evidence
materially supports an unresolved need in that state. The same `0-9`,
`30-39`, `60-69`, and `90-99` bands encode zero through three useful
documents.

Both interventions are scored in one physical response, preventing
separate-run provider nondeterminism from confounding alignment.

## Frozen Gates

All gates must pass:

- exact 10 physical calls, zero reasoning, forced exits, retries, malformed
  objects, or repairs;
- all ten aligned/shuffled state pairs are distinct;
- blinded labels contain exactly five aligned-A and five aligned-B rows;
- aligned scores vary on at least 8/10 rows;
- aligned pairwise accuracy is at least `.60`;
- aligned choices are oracle-optimal on at least 7/10 rows;
- aligned-minus-shuffled pairwise accuracy is at least `+.10`;
- aligned has at least two more oracle-optimal choices than shuffled;
- aligned selected exact required-document total exceeds shuffled by at least
  two; and
- adapter-attributed cost is at most `$0.25`.

Exact values are used only after responses to score the blinded interventions.
No threshold, shuffle, label assignment, evidence length, prompt, parser, or
row may change after outcomes.

## Interpretation

Passing establishes that correctly aligned path-dependent generated beliefs
causally improve continuation ranking when they are the scorer's semantic task
representation. It authorizes only a separately frozen old-tree development
stage for a full root-plus-continuation policy.

Failure means either the generated states are not sufficiently informative or
the belief-only scorer cannot use them reliably. It does not authorize adding
the opening, queries, or other bypass channels back into this interface.

## Budget

Projected cost is below `$0.15`; hard cap is `$0.25`. The current live balance
is `$46.563267126`, so the protected `$25` reserve remains intact. OatML
remains paused.
