# Bongard Luna Shuffled-Control Amendment

Date frozen: 2026-08-06, before any Bongard model request or endpoint access.

This amendment corrects the compute-matched shuffled control in the full-tree
mechanics and development protocols. It does not change the primary
`dynamic_depth2` or `myopic_width` policies, task split, model, prompts, query
budget, endpoints, or success thresholds.

## Invalidated Definition

The previous implementation rotated branch-support objects across first
actions and then recomputed the best second query using the target action's
remaining-candidate set. If target action A received the branch generated
after source action B, this made already-observed B eligible for a second
query and removed still-unobserved A. The resulting score did not preserve
valid query eligibility or the quality distribution of complete continuation
values. No model request or endpoint existed under that implementation.

## Frozen Corrected Control

For each real first action `a`, compute its complete expected continuation
value using its own regenerated branches and valid remaining candidates:

```text
V(a) = P_root(+|a) max_{b != a} EIG(branch[a,+], b)
     + P_root(-|a) max_{b != a} EIG(branch[a,-], b)
```

Let `pi` be the same one-position derangement of sorted opaque candidate IDs.
The corrected shuffled score is:

```text
shuffled_score(a) = root_EIG(a) + V(pi(a))
```

Thus the multiset of complete expected continuation values is preserved
exactly while its action coupling is broken. No branch support is evaluated
under another action's candidate-eligibility set. After the shuffled control
selects its first action, execution still uses that action's real regenerated
branch, matching the primary policy execution contract.

Every task records the real and shuffled continuation-value maps and the
derangement. Mechanics and development blocks require the shuffled values to
be an exact permutation under that mapping.

## Decision Robustness

Every nonrandom policy now records its selected-score margin over the runner
up. A dynamic-versus-myopic action change counts toward the mechanics or
development diversity gate only when the dynamic score advantage over the
myopic-selected action is at least `1e-6` nats. This excludes numerical ties
without imposing a scientifically meaningful effect-size threshold.

Interface versions are bumped to mechanics `-3` and development `-3`.
Serving remains `-2` because its belief-generation contract is unchanged.
