# ChemBench Posterior-Sampling Fidelity Result

Date: 2026-08-15 (Europe/London)

## Binding

- Result: `results/nonmyopic/chembench_posterior_sampling_fidelity/result.json`
- Result SHA-256:
  `d71964ff247bc80408b0b5c78c4b168c5117ece372446990c4a5dfd17b42c0f6`
- Protocol SHA-256:
  `b740e2a31b4f5ecc57b07661fb47b4f48bd173ca3aff814be735a1089a3bd542`
- Implementation SHA-256:
  `cc1ececf50ef0931e8aea0e402c68b6cabb704249d1dc744e978c4a62c7b6a66`
- Pooled predecessor SHA-256:
  `637c031946268e8015687d5eb4af4d037bfc1399300e56c0e50bdf049b4fab07`
- Component reference SHA-256:
  `e4533d76af6ab9be344bbf72806762eb695a425e7664a272105b3d075f706b16`

All 36 cases and every saved 2,048-outcome action-value replay completed in
30.74 seconds. No LLM, API, network call, endpoint, or paid resource was used.

## Frozen Decision

The gate **failed**. IID posterior sampling at 256 outcomes per action does not
open MCTS depth.

| 256 replicate | Median rho | Fraction rho >= .8 | Regret <= 3% | Mean regret |
|---:|---:|---:|---:|---:|
| 1 | 0.7473 | 44.44% | 97.22% | 0.211% |
| 2 | 0.8198 | 55.56% | 100% | 0.154% |
| 3 | 0.6945 | 38.89% | 100% | 0.072% |
| 4 | 0.8000 | 50.00% | 100% | 0.125% |
| Required, each | >= 0.90 | >= 90% | >= 90% | <= 1% |

Every 256 replicate passed the pooled and both component-bank regret gates.
Their mean pairwise selected-action agreement was 75.93%.

The non-gating 1,024-sample four-replicate ensemble reached median rho 0.9033,
but only 77.78% of cases reached rho 0.80. Its pooled top-action regret was at
most 3% in every case with mean 0.0557%; component-bank gates also passed.

## Interpretation

The estimator usually identifies a practically good action but cannot rank the
full 14-action set faithfully at a usable IID sample budget. This is not fixed
by simply aggregating four times the planned compute. Meaningful occasional
misses remain, such as 4.80% normalized pooled regret for medium cooperative
inhibition in replicate one.

The failed first link blocks naive posterior-sampling MCTS. The next numerical
test should use randomized quasi-Monte Carlo over the two predictive random
variables: posterior particle CDF coordinate and standard-normal observation
noise. Scrambled low-discrepancy prefixes preserve the 32/64/128/256 power-of-
two ladder and support independent randomized replicates. This is variance
reduction, not a threshold, posterior, action-panel, or endpoint change.

If 256 randomized QMC samples fail the same gates, reduce the action set through
a prospectively frozen numerical shortlist before attempting depth. Do not
spend on an LLM to compensate for a noisy acquisition estimator.
