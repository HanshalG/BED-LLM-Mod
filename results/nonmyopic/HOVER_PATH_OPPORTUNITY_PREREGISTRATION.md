# HoVer Path-Dependent Retrieval Opportunity Preregistration

## Question

Does fresh HoVer contain enough exact, target-blind retrieval structure for a
non-myopic LLM policy to improve supporting-document coverage through its own
path-dependent semantic belief updates?

This is a zero-model-call substrate audit. It does not test an LLM and cannot
support an efficacy claim. Its purpose is to reject HoVer before paid work if
the apparent many-hop structure does not induce a real first-action planning
gap.

## Frozen Inputs

- HoVer semantic-BED manifest SHA-256:
  `d64db625ca142b97df990b1831b62c1d4438b1f52555c09388e2f1762f733b13`.
- Exactly the 400 opportunity UIDs in manifest order: 200 three-hop followed
  by 200 four-hop.
- HoVer dev source SHA-256:
  `67c14858f2d7fcdb96b6fe3d538ffcd6f76e3ba594aa2c0cd4359f601101e89d`.
- Official dev TF-IDF retrieval artifact SHA-256:
  `b50a961f63a95ff184986af766b15ff6b1d6c98f7e86b39355b64b1a85fb3745`.
- Official Wikipedia SQLite database SHA-256:
  `c37ee397916ec0bffacfe8902db454a5cda88a7a188409217b2e15231fe5ee2f`.
- Candidate ordering is the official top-100 TF-IDF title ordering for each
  claim. The first ten titles are legal root actions.

Development and holdout rows remain sealed. No claim, label, support title, or
support sentence from those splits may be read.

## Frozen Transition

Opening a document returns its exact text from the official SQLite database.
For a current document, a continuation title is legal when either its full
normalized top-100 title or its normalized title with one trailing
parenthetical removed appears as a complete token phrase in the document.
Single-token aliases shorter than five characters are excluded. The current
title is excluded and documents cannot repeat within a path.

This transition is target-blind:

- it uses only the retrieved article and the frozen top-100 title pool;
- it does not use support annotations to create links;
- it does not expose a menu to a future LLM policy; and
- it can be replayed exactly without live web search.

The audit enumerates legal paths of at most three opened documents from every
root. A path's external value is the number of distinct HoVer supporting
document titles it covers. Sentence indices and verification labels do not
affect selection.

## Frozen Comparisons

For each task:

- `immediate(root)` is one when the root is a supporting document and zero
  otherwise.
- `V2(root)` and `V3(root)` are the maximum exact support coverage reachable
  from that root with at most two or three total documents.
- `rank_myopic` maximizes immediate coverage and breaks ties by official
  TF-IDF rank. It receives its own oracle depth-3 continuation tail.
- `robust_myopic` is stronger: among *all* immediate-optimal roots, it selects
  the one with the best oracle depth-3 tail, then breaks ties by rank.
- `oracle_d2` maximizes `V2`.
- `oracle_d3` maximizes `V3`.
- `rank_root_gain = V3(oracle_d3) - V3(rank_myopic)`.
- `robust_root_gain = V3(oracle_d3) - V3(robust_myopic)`.

A rank-sensitive opportunity has positive `rank_root_gain` and a changed root.
A robust sacrifice opportunity has positive `robust_root_gain` and strictly
lower immediate coverage at the depth-3 root than at the robust-myopic root.
The robust definition prevents arbitrary myopic tie-breaking from creating the
headline opportunity.

## Frozen Gates

All conditions are conjunctive:

1. all source, manifest, retrieval, and database hashes match;
2. exactly 400 opportunity rows are analyzed in manifest order;
3. at least 390 rows have 100 distinct, database-resolved candidates;
4. at least 300 rows contain a nonempty legal continuation graph;
5. at least 250 rows have nonzero depth-3 support reachability and depth-3
   root-value range of at least one;
6. at least 40 rank-sensitive opportunities exist, including at least 15
   three-hop and 15 four-hop rows;
7. at least eight robust sacrifice opportunities exist;
8. at least 20 rows have strictly greater best depth-3 than depth-2 coverage;
9. mean `rank_root_gain` is at least `0.10` supporting documents;
10. mean `robust_root_gain` is at least `0.02` supporting documents.

The robust thresholds correspond to at least a 2% exact population of genuine
first-action sacrifices in this 400-row screen. The rank-sensitive gates retain
the broader planning opportunity needed for a practical task pool.

## Decision Rule

A full pass authorizes only a separately committed, low-cost mechanics smoke
on preregistered development rows. That smoke must make the LLM generate and
regenerate semantic evidence hypotheses; exact graph enumeration is an oracle
and may not become the deployed method.

Any failed gate closes this exact HoVer interface. Do not change root width,
path depth, title normalization, thresholds, task subset, or retrieval source
after viewing the opportunity result. No OpenRouter call or OatML job is
authorized by this preregistration.
