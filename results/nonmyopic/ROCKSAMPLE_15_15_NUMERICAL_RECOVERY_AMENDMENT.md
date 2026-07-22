# RockSample[15,15] Numerical Recovery Amendment

Registered 2026-07-22 after the first formal attempt failed and before changing
the updater or launching a replacement attempt.

## Failure

The frozen paid run stopped without writing a policy endpoint after 859 requests.
The exception arose in the exact Bayesian updater:
`cannot update on an impossible Rock Diagnosis observation`. The provider had no
parse or terminal failure, and the project ledger records `$0.36174112` for the
incomplete attempt. A full 30-trial, 15-round, 1,320-cell deterministic execution
under the same map, seed, arms, and concurrency completed all mechanics, so this
is a rare-history numerical path rather than a general geometry or memory failure.

The updater currently classifies every predictive normalizer at or below `1e-12`
as impossible. A strictly positive finite normalizer is a valid, possibly rare,
Bayesian observation and must be normalized rather than rejected. Fifteen rounds
and high-accuracy sensors make such values reachable. True impossibility is zero
mass (or a non-finite computation), not small positive mass.

## Frozen Repair

- Change only the posterior guard from `normalizer <= 1e-12` to rejecting
  non-finite or `normalizer <= 0` values.
- Add a regression test where an observation has positive predictive probability
  below `1e-12`; its posterior must remain finite, normalized, and exact.
- Retain the map, model, non-thinking mode, temperature, seed `24100`, K4, h2,
  30 paired trials, 15 rounds, controls, bootstrap count, concurrency, endpoints,
  prompts, scorer, and all original gates unchanged.
- Rerun the complete formal stream because the exception occurred outside the
  provider's failed-closed cache writer and no accepted-cell artifact exists.
  The incomplete attempt has no policy endpoint and will never be pooled.
- Keep the original `$1.50` aggregate run cap. The failed attempt and replacement
  share the same run id and ledger entry, so the cap includes both automatically.

This amendment repairs only the mathematical definition of possible evidence; it
does not inspect or optimize a StrategyEIG endpoint.

## Replay Status

The first replacement attempt reached OpenRouter but was rejected immediately
with HTTP 402 before producing an endpoint. The provider credit endpoint reported
`$40.00` total credits and `$40.208810965` total usage. This is an external account
balance, distinct from the repository ledger's `$30.59453806` tracked usage. The
unchanged replay remains pending a provider top-up; the 402 attempt is not a run.
