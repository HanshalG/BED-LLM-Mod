# Partial chance-bound search

Code/audit frozen and pushed at 5e7cc74a before a single public-panel run.
The new use_chance_bounds flag defaults false and requires adaptive action bounds
and a valid state_risk_lower_bound hook. The state bound is the weighted
known-family optimum on the full repeatable menu. The lower-bound induction for
the horizon-corrected positive quadrature operator supplies its justification.

For an action with a finite feasible incumbent, compute all branch lower bounds
and the fixed signed quadrature correction. Evaluate branches in descending
probability order, substituting exact continuation contributions. Abandon only
when their sum with remaining lower contributions and correction exceeds the
incumbent by the existing conservative margin. Return that total explicitly as
a pruned root bound when applicable, never as an exact value or cached optimum.
Terminal batches still charge every evaluated leaf. No action, family, physical
measurement, quadrature node or runtime cap is removed or changed.

## Tests and audit

211 focused tests pass in 22.75 seconds; scoped E4/E7/E9/F lint passes. The full
tree/exhaustive comparisons now cover both pruning modes. A separate deterministic
fixture forces an actual partial abandonment with a negative correction: saved
bound 4.5, exhaustive action value 5.5. The chosen tree matches exactly, fewer
nodes are evaluated, and a malformed state bound fails closed.

Artifact SCILAWS_PARTIAL_BOUND_SEARCH_AUDIT_20260908.json SHA256:
7e71d8733bba0baba31b72626cc26be008762d16427bcdb6ecfc9560b4063fcb.

Same eight public tasks, order8, five seconds and100000 nodes per plan:

- All h1/h2 cases complete. H2 takes .642-.709 seconds.
- H2 nodes: baseball3169 unchanged; bird/spirometry4555 versus5281;
  one-dimensional cases10360 versus13249. Partial cut counts are0/1/3
  respectively, including materialization work.
- All completed root choices and risks match preceding bound-only results to
  1e-12 in an offline saved-result comparison.
- H3 still fails8/8: baseball max_seconds, seven others max_nodes. No completed
  depth3 policies or risks, no automatic rerun, no cap relaxation.

This is a modest search improvement, not a solved runtime problem or source-world
result. Same-dimensional priors are repeated mechanics, not independent efficacy.

## Next decision

Stop treating increasingly elaborate pruning as sufficient. The remaining
numerical question is integration accuracy as well as cost. Prospectively audit
orders4/8/16 on nonzero-history mixed-family analytic-working-model cases with
the same complete menus, explicit all-root values, and unchanged accuracy
threshold. Compare against refined integration on tractable cases, including
action regret and family-bound consistency; single-family exactness alone is
inadequate. A lower-order rule is only a candidate, not authorized for deployment
because it runs faster. If it fails, bank that failure rather than change the
threshold or silently drop difficult cases. Numerical qualification does not
establish source calibration or license a paid proposer experiment.

Source usage review, sealed measurements, useful LLM proposal evidence, paired
depth comparisons and anticipated discovery remain unfinished. No source
measurements/modelcalls; $0. Authenticated credits/usage/balance remain
245/220.376693994/24.623306006, Sept8 London spend0. Process exited; automation
paused; goal active and incomplete.
