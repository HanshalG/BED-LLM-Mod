# Animals Stratified Prior Target Result

Status: **sampler passed; target pool fixed before policy responses**.

Eight taxonomic mixture-component calls produced 127 unique cleaned names and
126 unique validated animals. Every stratum returned 15 or 16 cleaned names.
Shuffle seed `24285` fixed:

- 20 development targets;
- 60 untouched confirmatory targets;
- 46 permanently unused targets.

No policy, belief update, candidate score, or endpoint was evaluated while
creating the pool. The pool is now immutable, and policy prompts must never
receive it.

Usage: 135 requests including validation, 8,456 prompt + 716 completion tokens,
zero reasoning, `$0.00104036`. Project spend is `$44.16668202` of `$110`.

Artifact:
`results/nonmyopic/animals_stratified_prior_targets/gemma26b_seed24285/TARGET_POOL.json`.
