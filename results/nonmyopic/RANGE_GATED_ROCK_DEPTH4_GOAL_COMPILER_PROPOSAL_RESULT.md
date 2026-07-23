# Range-Gated Rock Hierarchical H4 S1 Result

The preregistered proposal gate failed closed before exact scoring. It is a
serving failure, not a policy-quality endpoint.

## Frozen Setup

- Architecture: Gemma 4 26B thinking target assignment, deterministic
  shortest-path compiler, exact h4 verifier.
- Fresh seed: `24229`.
- Planned cells: 16 distinct strict h4 opportunities.
- Producer/audit bootstrap seeds: `24230/24231`.
- Preregistration:
  `RANGE_GATED_ROCK_DEPTH4_GOAL_COMPILER_PREREGISTRATION.md`.

## Failure

Fifteen cells returned legal four-target assignments. Cell 11 returned the
provider notice

```text
[Wafer: response was truncated before the model finished its internal reasoning. ...]
```

on both its initial response and the single registered correction. Because this
non-JSON notice is nonempty, the adapter did not classify it as an empty
reasoning-only length stop and did not invoke its forced-final continuation for
that cell. The all-cells gate therefore failed before scoring.

No alternate seed, prompt repair, adapter repair, replacement cell, or S1 rerun
is used.

## Prefix Diagnostics

- Accepted logical cells: 15/16.
- Invalid cell: 11, both registered attempts.
- Rock 4 assigned to the fixed north root: 13/15 accepted cells.
- All accepted assignments had four distinct valid targets.
- Formal random-goal, shared-h3, strong-d3, recovery, and route-selection
  endpoints: not scored.

The 13/15 prefix is consistent with the passed 10/10 S0 mechanism, but it is not
a preregistered proposal endpoint and does not support an h4 policy claim.

## Usage

- Physical requests: 31.
- Prompt tokens: 43,456.
- Completion tokens: 67,953.
- Reasoning tokens: 46,070.
- Forced exits: 16.
- Forced-final requests/successes: 14/14.
- Cost: `$0.02883590`.
- Project spend after S1: `$40.29151733245984 / $110`.

## Interpretation

External route compilation removes the demonstrated low-level spatial
composition error: the model selected the critical semantic target-root pairing
reliably in S0 and most accepted S1 cells. Sustained provider finalization remains
load-bearing. The frozen h4 goal-compiler line stops without a proposal or
trajectory claim. The standardized Wafer-notice path may be repaired generically
for future experiments, but never used to relabel or rerun this seed.
