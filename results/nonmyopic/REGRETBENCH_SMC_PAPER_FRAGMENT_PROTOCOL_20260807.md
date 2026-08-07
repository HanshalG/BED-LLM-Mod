# RegretBench SMC Paper Fragment Protocol

Date frozen: 2026-08-07, before any RegretBench SMC policy model response.

Status: sealed deterministic paper mapping. It changes no experiment or current
manuscript claim while the generated fragment is absent.

## Bound Input

The fragment generator must recompute the SMC frozen report with generator
SHA-256
`2996d0cfd18e2ade3a7947492e0f07f13b1e452ee7d57a5cfbda27659bb11d43`
from the raw run and independently replayed verification. It must then require
the saved `FROZEN_REPORT.json` to match that recomputation exactly. Hand-edited,
partial, stale, or unverified reports produce no fragment.

## Deterministic Text

The fragment must:

1. identify the result as development-only;
2. state that DeepSeek annotates semantic reply likelihoods and performs
   path-dependent two-through-six retain/revise SMC transitions over exact
   banked parent slots;
3. print the exact frozen tier interpretation without paraphrasing its
   scientific strength;
4. use `smc_myopic_refresh_brier` as the headline horizon-isolating control;
5. show all six paired controls when mechanics pass and no efficacy table when
   mechanics fail;
6. report paired Brier difference, sample standard deviation, 95% interval,
   improvement probability, wins/ties/losses, and root disagreement;
7. report refresh-matched predicted-to-realized Spearman fidelity; and
8. label alignment-complete subsets, branch-draw stability, fresh final SMC
   regeneration, and optional Luna thinking as non-rescuing descriptive
   diagnostics that cannot alter the tier.

The `passed` development tier must say provisional and confirmation required;
it can never say confirmed. The `gated_null` tier forbids confirmation. The
`mechanics_failed` tier reports no policy efficacy.

## Manuscript Boundary

The default output is `paper/generated/regretbench_result.tex`, the same
already-conditional RegretBench include used by `paper/main.tex`. The pre-result
manuscript SHA-256 is
`6ece61e00c284c961e08375b873a410a959bf9bab978f847c0d099dbaf7453bf`.
The generated fragment is absent at freeze, so this protocol changes no current
paper text or page count. Only the deterministic verified fragment may occupy
that path.

Freezing and testing this mapping makes zero model calls and costs zero dollars.
