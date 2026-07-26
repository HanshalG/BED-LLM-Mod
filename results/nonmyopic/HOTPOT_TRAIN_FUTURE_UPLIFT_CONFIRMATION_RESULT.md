# HotpotQA Train Future-Uplift Confirmation Result

The single preregistered confirmation attempt failed closed before scientific
scoring and will not be repaired or rerun.

- run:
  `hotpot-train-future-uplift-confirmation-20260726T050858Z`;
- public artifact:
  `results/nonmyopic/hotpot_train_future_uplift_confirmation/hotpot-train-future-uplift-confirmation-20260726T050858Z/CONFIRMATION_FAILURE.json`;
- confirmation cohort endpoints materialized: exactly `500`;
- frozen scientific tasks selected: first `10` qualifying records;
- successful stages: `10` initial belief/root-score calls and `40` branch
  refresh calls;
- failure: one refresh response ended mid-JSON at character `1,798`;
- physical requests and HTTP attempts: `50` and `50`;
- retries, reasoning tokens, and forced exits: all `0`;
- cost: `$0.4178325`, below the frozen `$1.20` run cap.

The strict no-repair parser gate stopped execution before any continuation
scorer call, policy selection, support-coverage endpoint, or final-answer call.
Consequently this run provides no scientific comparison among future-uplift,
myopic, old-total, fixed, shuffled, or random policies. The earlier opportunity
and serving passes remain structural and mechanics evidence only.

The confirmation split is now consumed. The 100-record development split and
71,391-record holdout remain endpoint-sealed. Combined serving and confirmation
spend was `$0.49679`, leaving `$1.28170905` of the frozen research allowance.
