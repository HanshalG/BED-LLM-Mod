# Split-resolution instrument and cost preflight

Diagnostic only: score_h2 holds posterior families, likelihood, actions and targets
fixed, uses the outer model's chance nodes/correction, and an independent corrected
model for final-step integration. It does not mutate either model or produce a
deployable policy tree. Real and simulated posterior updates remain unchanged.
Default production planner is untouched.

Equal-order complete roots match the existing planner on all four previously
declared two-family histories to1e-12. Unequal4/8 matches a separately assembled
sum of branch-wise optimized terminal risks. Preflight rejection is tested before
branch generation. All14 split/horizon-correction tests pass in1.76s; scoped lint
passes. These are software equivalence checks, not independent integration accuracy.

For A actions, F nonzero component rules and per-level orders q_i, the unmerged
terminal leaf count is product(A*F*q_i). This excludes internal states, numerical
moment-correction arithmetic and policy-tree materialization; it is not a timing
estimate. Exact node merging can reduce work. The diagnostic conservatively refuses
before evaluation when 1+A*F*q_outer+A^2*F^2*q_outer*q_inner exceeds its node cap.
Actual terminal leaves are still charged even when vectorized. No per-action reset.

With A8,F4:
- h2 outer16/inner4: 65536 leaves,66049 diagnostic nodes including internal states.
- h2 outer8/inner8: 65536 leaves,65793 diagnostic nodes.
- h2 outer32/inner4: 131072 leaves, already over100000.
- h3 orders2/2/2: 262144 leaves before internal work.

Consequently unequal resolution alone cannot deliver unpruned full-geometry h3
under this cap. A full h3 implementation needs verified pruning or qualified
value-function approximation, not faster evaluation or relabelled leaf counts.
No claim of depth monotonicity follows from this arithmetic or these tests.

Next dependency: prospective independent terminal accuracy check on actual
outer-branch posteriors before any mixed-resolution h2 comparison. Cost-preflight
that diagnostic first. Do not infer future-state accuracy from h1 root agreement.
Use the resulting evidence to decide whether full-geometry value reuse is feasible;
do not repeat a predetermined over-cap tensor sweep. All prior numerical failures
remain banked. Source calibration, useful LLM proposals, receding-horizon paired
episodes and anticipated discovery remain unfinished.

No source measurements, LLM calls or spend. Live account usage220.376693994,
balance24.623306006; London daily ledger spend0. Automation remains paused.
