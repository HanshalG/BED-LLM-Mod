# Animals Implicit-Prior Target Protocol

Status: **frozen before target-sampler responses**.

The hand-curated exact-name holdout placed most targets in the far tail of the
LLM's implicit animal prior, making branch recovery nearly impossible. A
Bayesian design benchmark should draw latent targets from its stated prior.

This distinct protocol samples a fixed benchmark pool before any policy
response:

- non-thinking Gemma 4 26B;
- 16 independent unconditional calls at temperature `1.0`;
- up to 16 animal names per call;
- unchanged structural cleanup and animal-name validation;
- case-insensitive deduplication;
- deterministic shuffle seed `24284`;
- first 20 names become development targets;
- next 60 become untouched confirmatory targets;
- remaining names are unused.

The sampler passes only if it yields at least 80 unique validated names in one
attempt, with zero reasoning and cost below `$0.25`. No replacement calls,
manual name editing, or endpoint-conditioned target selection are allowed.

The target pool is never included in policy, belief-generation, likelihood, or
ranker prompts. It defines the latent prior support for evaluation only.
