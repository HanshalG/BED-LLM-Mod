# ChemBench costed-repeat CRN evaluation amendment

Date frozen: 2026-08-15

This amendment is frozen before any v4 response matrix is constructed. It
supersedes only the phrase "exactly integrates the three categorical outcomes"
for **full eight-well execution evaluation** in
`CHEMBENCH_COSTED_REPEAT_CORRIDOR_PROTOCOL_20260815.md`.

The policy decision itself remains exact within the frozen action shortlist: every
d1, d2, or d3 lookahead integrates all three outcomes at every planned branch.

## Reason

An eight-decision r=1 execution tree has `3^8` leaves. Exact replay is unnecessary
for comparing policies and makes the eight random controls disproportionately
expensive. Paired common random numbers provide a scalable unbiased policy-value
estimate without changing policy actions or exposing endpoint labels.

## Frozen execution evaluation

For each difficulty and each of its nine held-out truth particles, generate 128
vectors of eight independent `Uniform(0,1)` values. At decision position `t`, map
the shared uniform through the selected action's exact truth-likelihood CDF to
sample its categorical outcome. Continue until no feasible action remains.

All primary depths and all controls use the identical uniforms for a given
difficulty, truth, scenario, and decision position, even when they choose different
actions. Terminal target loss is deterministic conditional on the resulting
history.

Seeds are:

- easy: `2026084501`
- medium: `2026084502`
- hard: `2026084503`

Report:

- mean terminal loss for each truth over 128 scenarios;
- aggregate mean over the 27 truth cells;
- scenario-level standard error; and
- paired depth/control comparisons using truth-cell means.

All original gate thresholds and practical tie tolerance remain unchanged. A pass
cannot rely on changing scenario count, seeds, or replacing the paired means with
an unpaired estimate after results.

Immutable replay must reproduce every sampled outcome, action, and terminal loss.
