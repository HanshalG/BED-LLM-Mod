# Bongard-OpenWorld Luna Full Mechanics Amendment

Date frozen: 2026-08-06

Before any model request, history weights, prompt image roles, and branch
sensitivity were corrected by
`BONGARD_OPENWORLD_LUNA_SEMANTIC_VALIDITY_AMENDMENT.md`; interface `-1`
artifacts are invalid.

Before any model request, the shuffled control was further corrected by
`BONGARD_OPENWORLD_LUNA_SHUFFLED_CONTROL_AMENDMENT.md`. Mechanics interface
`-2` is invalid; the corrected interface is `-3`.

This amendment fixes the full four-task mechanics execution details before any
Luna image request. It does not change the exact-10 serving gate.

## Shared Call Tree

The semantic-validity amendment expands the final stage below to all eight
realized first-action continuations; its 16--40 final calls and `$1.75` cap
supersede the original selected-policy-only counts in this section.

After a passed exact-10 result, the original design generated:

- four root supports;
- 64 first-query supports: four tasks times eight candidates times two
  simulated outcomes;
- one final support per distinct realized `(task, six-label history)` selected
  by the five frozen policies.

The first stage is exactly 68 requests. There are at most 20 final histories,
so the total is `68 + distinct_final_histories <= 88`. Identical final
histories are generated once and shared across policies. Raw responses are
private and hash-bound; no prompt is repeated to make one policy look better.

## Frozen Policies

- `myopic_width`: root one-step EIG; after the actual first answer, one-step
  EIG under that action's regenerated branch support. It consumes the same
  shared branch cache as path-dependent depth two.
- `fixed_depth2`: cumulative two-step EIG under root support; second query from
  the analytically updated root support.
- `dynamic_depth2`: root EIG plus expected best second-step EIG under each
  regenerated branch; second query from the realized regenerated branch.
- `shuffled_dynamic_depth2`: rotate complete expected continuation values
  across first actions before root scoring; execute the chosen action using
  its real regenerated branch. This exactly preserves continuation-value
  quality and compute while breaking action-specific path coupling, without
  evaluating an observed image as a query candidate.
- `random`: two deterministic seeded candidates without replacement.

Ties are resolved by lexicographically smallest opaque image ID. The shuffled
mapping is a one-position rotation of sorted candidate IDs, with no fixed
points. Random seed: `2026081022` plus a hash-derived task offset.

Every policy receives exact released labels only for its two selected
candidates. Its final support is regenerated from exactly those six revealed
labels. Endpoint images and labels remain absent from every prompt.

## Mechanics Gates

All are conjunctive for `mechanics_pass`:

1. a hash-bound passed exact-10 Luna result and reconciled $5 daily ledger;
2. exact `68 + distinct_final_histories` accepted requests and HTTP attempts;
3. zero retries, provider errors, reasoning tokens, and forced exits;
4. every strict response parses to ten unique rules and 14 likelihoods;
5. all root, fixed, dynamic, shuffled, and second-step scores are finite and
   executable;
6. at least 24 of 32 candidate outcome-pairs are materially label-sensitive
   under the exact-10 definition;
7. `dynamic_depth2` changes the myopic first action on at least one of four
   tasks;
8. at least two of the four non-myopic/control policies produce a final
   history distinct from `myopic_width` somewhere in the four tasks;
9. all distinct final supports are generated once, parse, and map back to all
   five policies;
10. pooled myopic endpoint Brier is at least 0.03 or pooled myopic endpoint log
    loss is at least 0.15, preventing a saturated development endpoint;
11. all endpoint metrics are finite and use only the two official released
    endpoint labels per task;
12. public payloads contain no source UID/path/position, ground-truth concept
    or caption, or label-bearing filename;
13. cost is at most $1.50 and the account-wide daily ledger remains at or below
    $5 after reconciliation.

The semantic-validity amendment supersedes the stale `$1.50` text in item 13
with `$1.75`. The shuffled-control amendment additionally requires the
shuffled continuation values to be an exact permutation and requires at least
one dynamic action change to clear a `1e-6`-nat numerical-tie margin.

Endpoint Brier and log loss are descriptive mechanics outcomes. A failure or a
dynamic loss does not authorize tuning on confirmation tasks. It localizes the
next change to prompts, likelihood calibration, or path scoring on these four
mechanics tasks only.

## Development Boundary

Even `mechanics_pass` does not open the 32-task development partition. A later
prospective development protocol must be frozen from this mechanics result,
with paired controls and an external endpoint, before any development image is
sent to a model. The 64-task confirmation and 199-task sealed test remain
untouched throughout.
