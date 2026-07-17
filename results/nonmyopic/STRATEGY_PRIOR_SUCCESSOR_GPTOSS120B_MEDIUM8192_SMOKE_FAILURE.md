# gpt-oss-120B Medium/8192 Serving-Gate Failure

The corrected gpt-oss serving configuration returned a complete JSON response with
1,267 reasoning tokens and no forced exit. It nevertheless failed the first strict L1
cell because several strategy rules ended with conditional `step_at_least` clauses
instead of the grammar's required unconditional `when: []` fallback. No policy endpoint
was evaluated; the one request cost `$0.00036933`.

This establishes that the serving budget was viable but that this model did not satisfy
the no-repair strict grammar gate. Per the successor registration, it is swapped rather
than given a policy-run retry.

Machine-readable artifact:
`strategy_successor_gptoss120b_medium8192_interface_smoke/20260718/SMOKE_FAILURE.json`.
