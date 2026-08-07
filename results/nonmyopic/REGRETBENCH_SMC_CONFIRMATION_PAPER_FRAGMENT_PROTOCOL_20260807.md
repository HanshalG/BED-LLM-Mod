# RegretBench SMC Confirmation Paper Fragment Protocol

Date frozen: 2026-08-07, before any RegretBench SMC response.

Status: sealed deterministic mapping from an independently verified untouched
confirmation report to the manuscript. It changes no experiment, result, or
current manuscript while the generated fragment is absent.

## Bound Input

The generator must independently rebuild the frozen confirmation report from
raw parent-bank and policy artifacts. The stored `FROZEN_REPORT.json` must
match that reconstruction exactly. Missing, partial, edited, stale, or
unverified inputs produce no fragment.

## Frozen Text Tiers

- `smc_confirmation_mechanics_failure_no_result`: state that confirmation was
  inconclusive and the development result remains provisional; show no
  confirmation efficacy table.
- `smc_confirmation_null_no_headline_result`: state that the untouched
  confirmation did not satisfy the preregistered conjunction and the
  development signal is not a headline result.
- `smc_confirmed_nonmyopic_semantic_particle_result`: state that the untouched
  cohort confirms non-myopic depth-two planning over LLM-owned reply
  likelihoods and retain/revise semantic transitions.

No other tier or wording strength is permitted. Development and confirmation
must remain separate and may not be pooled for a claim.

## Mandatory Content

When mechanics pass, show all six paired controls with root disagreement,
Brier difference and sample standard deviation, 95% interval, improvement
probability, and wins/ties/losses. Identify `smc_myopic_refresh_brier` as the
headline horizon-isolating control and report its predicted-to-realized
Spearman diagnostic. State that the frozen tier uses the prospectively amended
13-gate conjunction for the refresh-matched and history-blind primary claims,
while all 34 original diagnostics remain visible and secondary gates can
neither veto nor rescue the tier. State that the hidden CIG never enters the planner, the
LLM owns semantic reply likelihoods and path-dependent two-through-six
retain/revise transitions, and alignment-complete subsets, draw stability,
fresh regeneration, optional baselines, pooling, and subgroups cannot alter
the tier.

## Manuscript Boundary

The output path is `paper/generated/regretbench_result.tex`, replacing any
provisional development fragment only after confirmation independently
verifies. The existing conditional include in `paper/main.tex` is reused.
The pre-result manuscript SHA-256 is
`6ece61e00c284c961e08375b873a410a959bf9bab978f847c0d099dbaf7453bf`.

Freezing this mapping makes zero model calls and costs zero dollars.
