# Invalid Pooled Replication Synthesis Attempt

Status: **invalid implementation output; excluded from scientific results**.

The first zero-call execution reconstructed the prospective cohort's
parent-only and generated-only roots from generic
`retained_parent_only_depth_three_root` and
`generated_only_depth_three_root` selection fields. Those fields belong to a
different retained-risk analysis and do not reproduce the confirmation's
frozen second-refresh ablation.

The mismatch was detected before banking:

- reconstructed parent-only root differences: `25/32`;
- frozen source parent-only root differences: `19/32`;
- reconstructed mean merged-minus-parent Brier: `-0.0012147`;
- frozen source mean: `-0.0014465`.

The invalid output is preserved as `INVALID_RESULT.json`, SHA-256
`20dcb1a115dfd6575905b612d97d7a8643a9c5d04f9ac8ad8dfa52a90e182ea4`.
It must not be cited or pooled.

The correction uses the exact hash-bound rows already stored at
`second_refresh.rows` in each source artifact and adds pre-aggregate
invariants requiring exact reproduction of source root counts and mean
effects. No model call, endpoint, tree, bootstrap seed, threshold, or
scientific analysis changed.
