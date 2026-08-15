# ChemBench Local-Tree Branch-Fidelity Result

Date: 2026-08-15 (Europe/London)

## Binding

- Result: `results/nonmyopic/chembench_local_tree_branch_fidelity/result.json`
- Result SHA-256:
  `e4533d76af6ab9be344bbf72806762eb695a425e7664a272105b3d075f706b16`
- Protocol SHA-256:
  `3bc5d2db761c0e7bb28dab397f155042f8ac5ce97511b2b7533a0ae79f782f8a`
- Implementation SHA-256:
  `b797f73b91251dcad929582602b96672be63a9af4a27b2d5a0809fb52b63cca4`
- Authorized V3 result SHA-256:
  `dce0832190aa8b76c9b345220a9043edb6e622db3b8cf0f524b68b5c06c783c1`

The run completed all 72 bank/case evaluations in 32.37 seconds with no model,
API, network, or paid call.

## Frozen Decision

The gate **failed**. Nine raw-observation quantile branches do not open local
depth development.

| Frozen condition | Bank 1 | Bank 2 | Required | Result |
|---|---:|---:|---:|---|
| Median action-risk Spearman | 0.9363 | 0.9538 | >= 0.90 | Pass |
| Fraction Spearman >= 0.80 | 80.56% | 80.56% | >= 90% | **Fail** |
| Fraction normalized regret <= 3% | 100% | 100% | >= 90% | Pass |
| Mean normalized top-one regret | 0.1633% | 0.0498% | <= 1% | Pass |
| Bank selected-action agreement | \- | 69.44% | >= 75% | **Fail** |
| Nine nonworse than five | Pass | Pass | Both | Pass |

Every compatibility, posterior, outcome, weight, and risk check was finite.

## Diagnosis

The failure is not primarily large top-action regret. Nine branches selected a
reference-optimal or near-optimal action in every bank/case under the frozen
3% threshold. Low full-list rank correlations often coexist with exactly zero
top-one regret, including hard Hill competitive, medium Hill Arrhenius, and
easy/medium allosteric-activation Arrhenius cases. In the easy allosteric case,
the complete action-value spread is only 0.29% or 1.76% of root risk across the
two banks, so ordering all 14 nearly tied actions is unstable and practically
weak.

There are still meaningful approximation misses: bank-one hard fractal
competitive has 2.32% normalized regret, and easy cooperative inhibition has
1.65%. The frozen rank-coverage and bank-agreement failures therefore cannot
be waived as ties.

The architectural mismatch is that equal-mass bins preserve scalar outcome
probability, not diversity of the posterior states induced by those outcomes.
A local Bayes-adaptive tree acts on child beliefs. The next prospective gate
should cluster or select branches in posterior-predictive belief space, while
retaining representative observations for residual-conditioned typed
mechanism proposals. It should also report a two-bank ensemble action, rather
than silently treating one finite-particle bank as ground truth.

No depth, LLM support transition, or benchmark endpoint is authorized by this
result.
