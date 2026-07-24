# Animals Implicit-Prior Target Sampler Result

Status: **sampler breadth gate failed; no policy run**.

The frozen 16-call non-thinking Gemma 4 26B sampler completed serving cleanly
but produced only 34 unique validated animal names, below the required 80.
Repeated stochastic calls converged on similar common-animal lists despite
temperature `1.0`.

No development or holdout target split was created, and no policy response was
requested. Per protocol, the attempt is not extended with replacement calls.

Usage: 50 total requests including validation, 4,384 prompt + 1,011 completion
tokens, zero reasoning, `$0.00077886`. Project spend is `$44.16564166` of
`$110`.

This rejects iid prompting as an adequate implicit-prior sampler. A distinct
future protocol must define a semantic mixture of strata and use the same
mixture for target sampling and belief generation.
