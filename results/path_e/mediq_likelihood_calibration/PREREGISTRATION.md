# MediQ Likelihood Calibration Gate

Registered before the first `factored_record` model call.

## Purpose

This gate tests whether the MediQ likelihood model has the direction needed for BED.
It is not a policy comparison and its endpoint accuracy must not be reported as efficacy.
The action and outcome sequence is frozen to all ten selected interactions from the final
Step 0 run:

`20260714T043855_mediq-step0-set-dedup-v2-eig-nonthinking-26b-seed1304`

The replay starts from each frozen initial A-D prior, recomputes likelihoods with the new
factorization, applies those likelihoods sequentially to the frozen observations, and
does not generate questions, call the patient, or decode a new answer.

## Registered Repair

`mediq_likelihood_mode: factored_record` replaces one joint three-category prediction
per option with two factors:

1. A label-independent distribution over whether the hidden original record explicitly
   answers the query.
2. An option-conditioned Yes/No distribution given that the record explicitly answers.

The second prompt states that MCQ options can be diagnoses, mechanisms, next steps, or
treatment priorities; they are not mutually exclusive descriptions of the patient.
Findings associated with another option may coexist. This targets the concrete failure
where high glucose and acidosis were treated as evidence against treating hypoperfusion
first.

The resulting unavailable likelihood is exactly equal under A, B, C, and D. Therefore an
unavailable response contributes no likelihood ratio and cannot move the posterior.

## Frozen Checks

The replay passes only if every check passes:

- exactly five tasks and ten turns are replayed;
- maximum unavailable-likelihood span across labels is at most `1e-12`;
- maximum posterior L-infinity movement on unavailable turns is at most `1e-12`;
- the realized available outcome has likelihood above its prior-predictive probability
  under the true label on at least 60% of available turns;
- mean true-label log-probability gain on available turns is strictly positive; and
- mean true-label log-probability gain over all turns is nonnegative.

The final two checks are directional diagnostics on only six available observations, not
confidence-interval claims. They are intentionally minimal: the legacy scorer favored
the true label on only four of ten total turns and produced mean truth-log gain of
`-0.2899` nats per turn.

## Decision Rule

- **Pass:** archive the replay, then run one separately held-out calibration smoke before
  freezing Claim 1. Do not count either calibration set in policy efficacy.
- **Fail:** do not launch Claim 1. Keep the finite A-D decode target, but replace the
  coarse option-only conditional with richer option-conditioned latent patient worlds
  or a separately registered data-estimation model. No prompt or threshold tuning on
  the same ten observations is allowed.

OpenRouter requests use Gemma 4 26B A4B, non-thinking, temperature zero, seed 1304. The
run config is `configs/config_mediq_likelihood_calibration_openrouter.yaml`; projected
spend is capped at `$0.03` against the authorized `$40` ledger.
