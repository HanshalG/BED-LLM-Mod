# Animals Branch-Generator Recall Screen

Status: **development model screen on inspected histories**.

The branch-content scorer is not identifiable when the generator almost never
recovers the hidden truth. On seed 24279, Gemma 4 26B included the target in the
union of six counterfactual branch supports for only `3/20` states.

This screen reuses the exact 20 target-free histories and three candidate
questions from seed 24279. It changes only the belief generator/checker to
non-reasoning GPT-5.4 Mini. Support capacity remains `16`, filtering remains
enabled, and every Yes/No branch uses the production update pipeline. Targets
are compared with returned supports only after all model calls.

The stronger generator passes only if:

1. all 20 states and 120 branches complete;
2. no target field enters any model-visible input; the prior support is the
   already generated target-blind support and may naturally contain the truth;
3. the target appears in at least `8/20` branch unions, materially above the
   fixed Gemma reference of `3/20`;
4. serving uses zero reasoning tokens and remains under `$2`.

Pass authorizes a separate scorer development run on these inspected records.
Failure rejects a model-only repair of the Animals open-world path.
