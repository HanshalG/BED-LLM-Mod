# ChemBench costed-repeat disk-audit runtime amendment

Date frozen: 2026-08-18

## Pre-amendment state

The scientific implementation is fixed at pushed commit
`d9559a2f03d415966200d9f74c3bd84bbe12f021`. The easy shard completed under
that implementation. The medium shard completed all policy optimization and is
still encoding its canonical audit after more than two days of wall time. It has
not written a shard result. The hard shard has never produced or exposed a
result. No medium or hard loss, root action, comparison, or gate value has been
inspected.

Process inspection localizes the runtime failure. The medium process holds about
38 GB of Python objects on a 16 GB host and pages through roughly 19 GB of swap
while the standard JSON encoder traverses millions of transition-audit records.
The records and proposal cache are integrity witnesses; they are not policy
state. Their in-memory representation is therefore an evaluator artifact, not a
scientific condition.

This amendment is frozen before the hard shard is opened. It does not authorize
interrupting the live medium process. If medium terminates without an atomic
result, its failure must be banked before the runtime below may be used to rerun
that missing shard.

## Frozen scientific identity

The source, v4 responses, held-out truths, inference particles, priors,
likelihoods, thresholds, actions, repeat counts, well budget, target seeds,
shortlists, proposal semantics, support transitions, policy ladder, CRN seeds,
scenario count, controls, losses, comparisons, and every gate remain byte-for-
byte those of scientific commit `d9559a2f`. The new runtime may not alter a
floating-point operation, candidate ordering, tie break, proposal key, policy
key, sampled outcome, action, or terminal loss.

## Disk-backed canonical mappings

Only `ProposalCache._cache` and each planner's `transition_audit` may use an
ephemeral SQLite mapping instead of a Python dictionary. Each table is keyed by
the existing 64-character proposal/audit SHA-256 key and stores the exact compact
sorted-key JSON bytes that the current canonical hash encoder consumes.

The mapping must:

1. implement exact key lookup and insertion without changing call order;
2. reject a duplicate key whose canonical value bytes differ;
3. expose keys in bytewise ascending order;
4. compute the existing mapping SHA-256 by streaming exactly
   `{json(key):canonical_value,...}` in that order;
5. decode proposal values as the original integer tuples;
6. preserve exact record, proposal, completeness, and truth-leakage counts;
7. use bounded SQLite cache and temporary files outside the result directory;
8. delete temporary databases only after their summaries or replay checks have
   completed; and
9. confer no authority on a partial database after interruption.

SQLite journal, synchronization, and cache-size pragmas may be chosen for an
ephemeral single-process workload. They are runtime settings only. An
interruption remains a failed attempt because no atomic shard JSON is written.

## Exact-equivalence gate

Before the hard shard may run, focused tests must compare the in-memory and
disk-backed paths on the same complete banks and require:

- identical canonical proposal and transition-audit hashes;
- identical proposal-cache hit/miss behavior and duplicate detection;
- identical d1, d2, and d3 result dictionaries;
- identical policy records, root actions, CRN outcomes, execution audits,
  truth-loss arrays, and replay hashes; and
- bounded resident-memory behavior for a synthetic mapping large enough to
  exercise streaming iteration.

Any mismatch closes this runtime. It cannot be repaired after the hard response
is opened without another prospective amendment.

## Mixed-runtime shard binding

The easy and, if it completes, medium shards retain their exact legacy binding
to `d9559a2f` and the five pre-existing protocol hashes. A disk-backed hard shard
must retain that same scientific binding and additionally record:

- the pushed runtime implementation commit;
- this amendment's SHA-256;
- `runtime_mode: sqlite_canonical_v1`;
- the exact-equivalence test manifest and hash; and
- zero model calls, network calls, and dollar cost.

The amended assembler may accept exactly this ordered combination: legacy easy,
legacy medium, and disk-backed hard. It must reject any changed scientific
binding, missing runtime evidence, unpushed runtime commit, reordered or duplicate
shard, or nonzero call/cost declaration. It calls the unchanged `apply_gate`
exactly once over the three unchanged slice payloads and reports both scientific
and runtime provenance.

This is a fail-closed runtime amendment only. It cannot rescue a failed
scientific gate, authorize an LLM call, open a policy endpoint, or support an
efficacy claim.
