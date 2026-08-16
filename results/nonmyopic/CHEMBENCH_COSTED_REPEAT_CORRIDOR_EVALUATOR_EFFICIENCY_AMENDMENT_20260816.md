# ChemBench costed-repeat evaluator-efficiency amendment

Date frozen: 2026-08-16

## Status before this amendment

The first execution of the pushed costed-repeat implementation was terminated
before writing any result after 13 hours. No slice, loss, root action, comparison,
or gate value was emitted or inspected. Process sampling showed that the runner
was materializing a very large canonical JSON representation of its proposal
cache after separately rebuilding the same exact planning tree for every depth and
again for immutable replay. This is evaluator redundancy, not a scientific
outcome.

The cohort, v4 responses, inference particles, observation model, thresholds,
assays, repeat counts, well budget, target seeds, shortlist, support transition,
CRN seeds and scenario count, controls, and all prospective gates in the original
protocol and CRN amendment remain unchanged.

## Exact cache sharing

Within one policy family and difficulty, d3, d2, and d1 are evaluated by one
planner instance. Its existing pure one-step-risk, policy-value, policy-action,
and proposal caches are shared across calls. Every policy key still contains the
full belief state, available actions, remaining wells, policy level, and candidate
action where applicable. Sharing a previously computed pure value must therefore
return exactly the value that an isolated instance would recompute. Transition,
predictive, and leaf-risk results are not retained in new unbounded caches because
the small recomputation cost is preferable to duplicating the full state graph in
memory.

A focused equivalence test must compare shared-instance and isolated-instance
outputs on a complete small bank before the full result may open.

## Streaming integrity digests

Canonical cache and audit digests are computed incrementally with the standard
JSON encoder using the same sorted-key, compact-separator representation as the
original implementation. This changes only peak memory: the byte stream and
SHA-256 digest must match the materialized canonical JSON digest on focused tests.

## Immutable replay without duplicate planning

The original protocol requires immutable proposal replay and the CRN amendment
requires reproduction of every sampled outcome, action, and terminal loss. It does
not require running the deterministic optimizer twice.

For every evaluated policy, the exact planner records each queried policy key and
selected action. Replay then:

1. reconstructs a fresh `ProposalCache` from the immutable oracle records;
2. exhaustively queries every recorded proposal key and requires the exact stored
   proposal, with no missing or extra record;
3. re-executes all 128 CRN scenarios for every held-out truth using the immutable
   policy-action records and the fresh replay proposal cache;
4. requires exact equality of selected actions, sampled outcomes, terminal losses,
   execution-audit fields, and their SHA-256 digests; and
5. requires every recorded policy key to be consumed and no unrecorded key to be
   requested.

This replay re-executes all support transitions and endpoint trajectories while
avoiding a second optimization over an already verified deterministic policy.
Fixed-support, full-support, cost-blind, and each random replicate receive the same
transcript treatment.

## Checkpointing and authority

The implementation may write one atomic result checkpoint per completed
difficulty and resume only when all protocol hashes, implementation commit,
settings, source binding, and completed checkpoint hashes match exactly. Partial
checkpoints have no scientific authority. Only the final ordered three-difficulty
result may be passed to the unchanged gate.

This amendment authorizes no model or network call and cannot rescue any failed
scientific condition. It exists solely to make the already frozen zero-call gate
finish with bounded memory and independently checkable replay evidence.
