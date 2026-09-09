# The correction prompts contained a measurable secondary-input signal

Retrospective public-input diagnostic only. Read all16banked request payloads
from correction_reasoning_comparison_20260909, not response reasoning, source
formulas or target outcomes. Request hashes and statistics are in
CORRECTION_PUBLIC_SIGNAL_AUDIT_20260909.json. No repeat model requests.

## Request integrity

Medium/high messages match exactly within each case/history condition. Every
payload names x0 and x1, supplies their public ranges and descriptions, and includes
observations explicitly labelled as log responses. Control histories are exactly
the first3rows of refreshed histories: x1 has1distinct value in the control and
4distinct values in the6-row refresh. Refreshed prompts include standardized
residuals; the missing variable is not omitted by message construction.

Subtract log(base(x)) from the six observed log responses, then fit an intercept
and a single x1 slope. The intercept absorbs global calibration. This is an
exploratory training-history calculation, not a new predictor scored on targets.

| Case | x1 slope | Constant-fit SSE | Linear-fit SSE | SSE reduction | Permutation diagnostic |
|---|---:|---:|---:|---:|---:|
| 0 | .219542 | .079540 | .018087 | 77.26% | 36/720 |
| 1 | -.303826 | .118270 | .000574 | 99.51% | 6/720 |
| 2 | .355188 | .285395 | .124543 | 56.36% | 72/720 |
| 3 | -.343655 | .159588 | .009013 | 94.35% | 6/720 |

Permutation diagnostic is the fraction of all720row permutations whose absolute
input-residual covariance is at least the observed value, retaining multiplicity
from repeated x1 values. It assumes exchangeability under a constant-residual
null. Six observations, varying x0, post-hoc choice of statistic and four inspected
cases prevent treating these numbers as a new confirmatory gate. The calculation
does not prove a correct functional form or held-out predictive performance.

## Interpretation

Missing x1 data is ruled out as the reason the banked models returned only constant
corrections. There was directionally structured public residual evidence to work
with. The audit does not uniquely identify reasoning, prompt, schema, or model
limitations, and does not establish a provider defect. The old medium/high null
remains closed. No retries, more effort, new seeds or rescue scores on its targets.

The interface demanded literal correction coefficients and then numerically
integrated only a global log scale. This mixes structural discovery with fitting
secondary coefficients from six noisy observations. A better role separation is
to request parameterized structures and let numerical code infer all coefficients.
That is an architectural hypothesis, not a demonstrated fix. The system prompt
already warned against constant multipliers, so simply adding that warning again
would not constitute a meaningful intervention.

The repo already implements this separation in
environments/chembench_mopen/structure_proposer.py, described in
CHEMBENCH_STRUCTURE_PROPOSER_INTERFACE_20260908.md. It accepts structure/parameter
names with experiment-owned priors and existing numerical inference. Reuse and
audit it when a new source gate is valid; do not build another redundant grammar.
Its prior tests are not proof that a real model supplies useful structures.

## Next evidence needed

A genuinely new source-conditioned structural proposal gate should measure the
LLM contribution after a competent parameter fitter, versus history-blind/shuffled
proposals and actual symbolic search using the same observations. Source-only
fixtures may qualify that interface but cannot become the BED headline. It must
precede independently calibrated future-answer/updater predictions and the full
same-budget ordinary-horizon comparison. Closed numeric cohorts and closed
ChemBench/NumberGame formulations are not authorized for reruns by this diagnosis.

This reorders work toward the causal bottleneck rather than another environment
adapter or higher reasoning budget. It does not adopt a new benchmark or authorize
paid calls without a prospective source/interface protocol. The full user goal
remains unchanged and unachieved.

Two tests pass in .09s: exact known linear relation and offset invariance, complete
permutation multiplicity, and constant-input rejection. Previous/current turns
progress. Calls0/cost0; authenticated usage221.306531939/balance23.693468061;
London-day conservative remaining4.11174654, prior .04 uncertainty retained.
