# MediQ Data-Estimation Calibration Gate

Registered before generating the held-out interaction bank and before the first
`data_estimation` model call.

## Why This Is A New Model

The rejected `joint_option` and `factored_record` models ask for
`P(patient finding | correct answer option)`. This is poorly defined for MedQA options
that denote a next step or treatment priority: several findings and answer-option
concepts can coexist. The failed frozen replay confirmed that a prompt warning did not
repair this conditional.

The new model uses BED-LLM's alternative data-estimation pairing because the conditional
target is now the small, explicit A-D space:

1. Elicit a label-independent marginal `r(y) = P(patient response y | history, query)`
   over Yes, No, and unavailable.
2. For hypothetical Yes and No, elicit
   `q(theta | history, query, y)` over A-D directly.
3. Set the hypothetical posterior for unavailable exactly to the current prior, making
   record missingness non-informative by construction.
4. Form the raw joint `r(y) q(theta | y)`, then use iterative proportional fitting on
   the Yes/No block so the final joint has exactly the current A-D prior as its row
   marginal and exactly `r(y)` as its response marginal.
5. Divide each joint row by its prior mass to obtain a proper likelihood table. EIG and
   deployed Bayes updates use this same table.

This projection prevents internally inconsistent hypothetical posteriors from silently
changing the current prior before an observation. It also makes the unavailable
likelihood exactly equal across labels. The implementation logs raw elicited marginals,
raw hypothetical posteriors, final likelihoods, and projection residuals.

## Held-Out Interaction Bank

The calibration bank is generated first by the belief-free, non-thinking naive policy
on usable official iMEDQA cases at offsets 5-14, three questions per case, seed 1304.
The data-estimation scorer is never called while producing this bank. Endpoint accuracy
is ignored. The bank is accepted only if all 30 turns map cleanly, are verbatim grounded,
pass relevance and action-contract checks, and then pass manual review. At least 15 of
30 outcomes must be available Yes/No; otherwise the calibration sample is inadequate
and no scorer conclusion is drawn.

The implementation and prompts are committed before this bank is generated. Cases 0-4
from the failed factorized replay are excluded and may not be used for tuning.

## Frozen Scorer Gate

After the bank passes its own validity review, replay all 30 frozen interactions once.
The data-estimation model passes only if every condition holds:

- exactly ten tasks and 30 turns are replayed;
- at least 15 outcomes are available Yes/No;
- maximum joint row/column marginal residual is at most `1e-10`;
- maximum unavailable-likelihood span and posterior L-infinity movement are each at
  most `1e-12`;
- the true label gives the realized available outcome more probability than the prior
  mixture on at least 60% of available turns;
- mean true-label log-probability gain on available turns is strictly positive; and
- mean true-label log-probability gain over all turns is nonnegative.

These are calibration-direction gates, not policy efficacy claims. No endpoint result
from the naive bank or replay enters the paper's method comparison.

## Decision Rule

- **Pass:** archive the calibration evidence, then freeze the paired Claim 1 methods,
  endpoint, cases, and analyzer before any efficacy run.
- **Fail or inadequate bank:** stop method changes and discuss the target/model mismatch
  with Hanshal. Do not tune prompts or thresholds on these outcomes and do not launch
  Claim 1.

Both runs use Gemma 4 26B A4B, non-thinking, temperature-zero structured judgments, API
seed 1304, and the pinned official MediQ data. Each config reserves `$0.08`; combined
expected spend is far below `$0.16` against the authorized `$40` ledger.
