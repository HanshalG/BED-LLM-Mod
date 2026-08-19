# ChemBench costed-repeat legacy-medium liveness clarification

Date frozen: 2026-08-19

## Observable state before this clarification

The legacy medium process is PID `23488`, started from exact pushed scientific
commit `d9559a2f03d415966200d9f74c3bd84bbe12f021`. It has emitted only the
nonnumeric markers that primary d3, d2, and d1 completed. It has not printed the
cost-blind marker or written `medium.json`. No medium action, repeat count, loss,
comparison, control, or gate value is visible.

Repeated stack samples place the process in the standard JSON encoder while
`_audit_summary` streams the cost-blind transition-audit hash. At freeze time it
has accumulated `326:28.81` CPU, occupies an approximately 38 GB physical
footprint on a 16 GB host, and pages through approximately 19 GB of swap at only
8-10% effective CPU. The terminal encoding phase has consumed roughly eleven
hours of wall time. This is a representation-liveness failure, not a scientific
result.

## Prospective cutoff

Continue preserving the exact legacy process until its accumulated CPU reaches
`330:00.00`. Immediately before acting, check for an atomic nonempty
`medium.json`. If it exists, do not signal the process; validate and use the
legacy shard under its original binding.

If no atomic shard exists at or beyond `330:00.00` CPU:

1. record PID, state, accumulated CPU, log SHA-256, output absence, memory/swap,
   and current time in a zero-call failure artifact;
2. send `SIGTERM` to PID `23488` and wait up to 30 seconds;
3. use `SIGKILL` only if the same PID remains alive after that grace period;
4. preserve the phase log and failure artifact permanently;
5. require the process and its PID-bound `caffeinate` assertion to exit; and
6. require memory pressure to recover before any replacement run.

The cutoff is fixed solely from runtime telemetry before any medium metric is
available. It may not be extended, shortened, or conditioned on a later partial
value.

## Replacement authority

If the cutoff fires, the missing medium shard may be rerun once with the pushed
`sqlite_canonical_v1` runtime from the disk-audit amendment. Its scientific
binding remains exactly `d9559a2f`; the replacement must use the original medium
query seed, CRN seed, responses, policies, controls, and gates. It must bind this
clarification, the disk-runtime amendment, the pushed runtime commit, and the
exact-equivalence manifest. It runs alone with a fresh empty runtime directory,
at least 6 GB free disk, and PID-bound `caffeinate`.

Hard remains unopened until an atomic medium shard has independently passed all
binding, replay, integrity, and finiteness checks.

## Assembly

The amended assembler accepts exactly one of these ordered runtime patterns:

- legacy easy, legacy medium, disk-backed hard; or
- legacy easy, disk-backed medium, disk-backed hard.

Every disk-backed shard must use the same pushed runtime implementation and
equivalence manifest. The assembler rejects disk-backed easy, legacy hard,
missing or duplicate difficulties, mixed disk runtime commits, altered
scientific bindings, nonzero calls/cost, or absent liveness provenance. It invokes
the unchanged scientific `apply_gate` exactly once.

This clarification changes no scientific response, policy, metric, control, or
gate. It authorizes no LLM/model/network call and cannot rescue a scientific
failure.
