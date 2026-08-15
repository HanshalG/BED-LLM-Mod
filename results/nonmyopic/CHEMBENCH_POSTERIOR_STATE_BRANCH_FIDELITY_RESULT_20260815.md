# ChemBench Posterior-State Branch-Fidelity Result

Date: 2026-08-15 (Europe/London)

## Binding

- Result: `results/nonmyopic/chembench_posterior_state_branch_fidelity/result.json`
- Result SHA-256:
  `637c031946268e8015687d5eb4af4d037bfc1399300e56c0e50bdf049b4fab07`
- Protocol SHA-256:
  `ac70e57c8dce66fb7911756ec6d3020e5eff97f4b07e717ec5e0ab19bdab145c`
- Corrected implementation SHA-256:
  `05330040923fef91c05d2d241a8a07725bdba69bcf916dddaa929cec181003bf`
- Predecessor result SHA-256:
  `e4533d76af6ab9be344bbf72806762eb695a425e7664a272105b3d075f706b16`
- Authorized V3 result SHA-256:
  `dce0832190aa8b76c9b345220a9043edb6e622db3b8cf0f524b68b5c06c783c1`

The complete rerun finished all 36 pooled cases in 224.23 seconds. It used no
LLM, API, network call, endpoint, or paid resource.

## Frozen Decision

The gate **failed** only the all-actions rank-coverage condition. Local depth
does not open.

| Condition | Posterior-state 9 | Raw-quantile 9 | Required | Result |
|---|---:|---:|---:|---|
| Median pooled Spearman | 0.9736 | 0.9560 | >= 0.90 | Pass |
| Fraction pooled Spearman >= 0.80 | 80.56% | 75.00% | >= 90% | **Fail** |
| Fraction pooled regret <= 3% | 100% | 97.22% | >= 90% | Pass |
| Mean pooled top-one regret | 0.0366% | 0.2206% | <= 1% | Pass |
| Bank 1 component fraction/mean | 97.22% / 0.6768% | \- | >=90% / <=1% | Pass |
| Bank 2 component fraction/mean | 97.22% / 0.7533% | \- | >=90% / <=1% | Pass |
| Nonworse than raw quantiles | Better | Baseline | Both metrics | Pass |

Every binding and finite/reproducibility condition passed.

## Interpretation

Clustering induced child beliefs is materially better than binning scalar
observations. It improves median ranking, raises rank coverage by 5.56 points,
and reduces mean pooled selected-action regret by 83.4%. The pooled action is
also practically robust to both calibrated component banks under the frozen
aggregate gates.

It is still not a faithful approximation of all 14 action values in enough
cases. Seven of 36 cases remain below rho 0.8. Some are nearly flat (easy
allosteric-activation Arrhenius has only 0.69% root-risk spread; hard
cooperative inhibition 0.52%), but others have broad value spread, including
medium Hill Arrhenius and easy sinh competitive. The frozen gate cannot be
rescued by the excellent top-one regret.

This closes fixed nine-branch local quadrature. The next architecture should
estimate values through posterior-sampled trajectories with common random
numbers and progressive widening, while retaining posterior-state summaries
for node merging and LLM proposal conditioning. A one-step sample-count
fidelity gate must pass before any depth experiment.

No depth or LLM support-transition claim is authorized.
