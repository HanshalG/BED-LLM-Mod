# Number Game Depth-Three Cross-Fit Audit

Date completed: 2026-07-28.

Status: **post-hoc development; zero model calls**.

## Question

The powered full-retention depth-three run improved average source-to-target
root ranking but did not beat depth two. This audit asks whether the remaining
failure is in-sample root-selection noise: the same generated source
hypotheses define both the simulated belief and the truths used to estimate
policy risk.

For each of the 20 open planning trees, the audit uses the next `k` target
supports cyclically as independent validation draws and preserves the tree's
own target support as the endpoint. No tree validates on its own endpoint.
Each draw receives equal weight, regardless of how many valid hypotheses its
LLM response contains. Depth three and depth two use exactly the same
validation draws; they differ only in whether risk is measured after three or
two adaptive queries.

## Sensitivity

| Independent validation draws | D3 vs D2 Brier gain | Wins / ties / losses | Root changes | D3 risk rho | D2 risk rho | Novel Brier difference |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.70% | 5 / 9 / 6 | 11/20 | 0.688 | 0.450 | +0.00574 |
| 2 | 1.21% | 7 / 11 / 2 | 9/20 | 0.781 | 0.483 | +0.00090 |
| 4 | 2.58% | 8 / 12 / 0 | 8/20 | 0.795 | 0.445 | -0.00577 |
| 8 | 3.20% | 10 / 9 / 1 | 11/20 | 0.850 | 0.446 | -0.00966 |
| 19 | 2.77% | 10 / 9 / 1 | 11/20 | 0.842 | 0.452 | -0.00636 |

At `k=8`, mean Brier is `0.154525` for cross-fitted depth three and
`0.159635` for cross-fitted depth two. The paired tree-bootstrap difference is
`[-0.00899, -0.00194]`. Depth-three pairwise concordance is `0.868`, versus
`0.675` for depth two. Compared with the original in-sample depth-three
selector, cross-fitting lowers Brier by `4.88%`, with 13 wins and no losses.

A shared local replay also shows a `3.05%` Hamming gain versus cross-fitted
depth two. Exact-extension coverage is `0.25` percentage points lower overall
and `0.32` points lower on novel targets. Coverage is therefore retained as a
reported mechanism metric, not used to define the fresh primary result.

## Interpretation

The broad depth-three simulator was not fundamentally anti-correlated with
the endpoint. Its failure was concentrated at the selected minimum: estimating
policy risk on the same finite LLM support used to construct the belief
transition overfit the argmin. Independent prior draws sharply improve
top-root fidelity while leaving the policy tree, prompts, and posterior
updates unchanged.

The `k` sweep and choice of eight draws use already-open endpoints, so this is
development evidence only. A separately frozen fresh 32-tree experiment is
required. Its held-out endpoint support is generated independently from all
eight validation supports.

Public result SHA-256:
`76b576f48d714d72d384917d2a5650787645f037ef01f43c4d8731cc6f10a0bd`.
