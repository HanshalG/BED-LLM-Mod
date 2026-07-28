# Number Game Cross-Fitted Depth-Three Pooled Audit Result

Date completed: 2026-07-28.

Status: **post-hoc pooled positive**.

## Scope

This zero-call audit pools the two independent 32-tree, 16-endpoint-draw
Number Game studies:

1. the fixed-policy fresh-endpoint study; and
2. the wholly fresh planning-tree replication.

The audit resamples 32 trees independently within each study for 20,000
stratified bootstrap replicates. It does not alter either study's
preregistered status, thresholds, policies, endpoints, or exclusions.

## Pooled Depth Result

| Policy | Trees | Mean Brier | Mean Hamming | Mean coverage |
|---|---:|---:|---:|---:|
| Cross-fitted depth three | 64 | 0.158118 | 0.073351 | 0.572979 |
| Equally cross-fitted depth two | 64 | 0.161906 | 0.075585 | 0.563704 |

Cross-fitted depth three improves Brier by `2.3393%`. The mean paired
candidate-minus-depth-two difference is `-0.003787`, with stratified
tree-bootstrap 95% CI `[-0.005862, -0.001975]`. It records 26 wins, 29
same-root ties, and 9 losses; the one-sided exact sign-test p-value excluding
ties is `0.002994`.

Coverage improves by `0.9275` percentage points, with 95% CI
`[0.1015, 1.7744]` points. Mean Hamming improves by `0.002234`, although its
95% CI `[-0.004768, 0.000213]` narrowly includes zero.

On novel targets, Brier improves by `0.005251`, with 95% CI
`[-0.008481, -0.002225]`, and coverage improves by `1.8972` points, with 95%
CI `[0.3310, 3.5658]` points. Novel-target Hamming is directional but its
interval includes zero.

## Ranking Fidelity

Independent-support ranking fidelity is consistently stronger at depth three:

| Ranking diagnostic | Depth three | Depth two |
|---|---:|---:|
| Mean Spearman Brier correlation | 0.9200 | 0.5078 |
| Mean pairwise concordance | 0.9146 | 0.7042 |

The stratified 95% intervals are `[0.8996, 0.9386]` versus
`[0.4237, 0.5871]` for Spearman correlation and `[0.8984, 0.9297]` versus
`[0.6702, 0.7372]` for concordance.

## Interpretation

The two independent studies estimate depth-three relative Brier gains of
`1.8619%` and `2.8070%`. Their descriptive effect difference is not resolved:
the index-paired interval crosses zero, and the studies use independent seeds.
The pooled estimate therefore strengthens robustness and precision but is not
a third confirmation.

The result supports the paper's first-link account. Once both policy selection
and evaluation use independent LLM-generated support draws, depth-three
planning ranks roots much more faithfully than equally cross-fitted depth two
and produces lower held-out posterior-predictive risk. The LLM remains
load-bearing in initial concept generation, both history-conditioned belief
refreshes, and independent support sampling.

Public source hashes:

- pooled audit `RESULT.json`:
  `ec2a2fb3f1e8da69ccc2863852abc9ef3ef907a817ec13cd375a4814de48a472`;
- fixed-policy fresh-endpoint `RESULT.json`:
  `47e684b11ac5c6340ff4f063ff7a14db8b8d0bd18b184d61f91e315903e4809e`;
- fresh-tree replication `RESULT.json`:
  `25e0939164f6806b30481e48a88372f22639f6065096cb59978600509d43a3d8`.
