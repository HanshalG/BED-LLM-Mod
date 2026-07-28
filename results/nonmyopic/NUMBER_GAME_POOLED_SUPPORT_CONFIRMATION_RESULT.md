# Number Game Pooled-Support Confirmation Result

Date: 2026-07-28

Protocol:
`NUMBER_GAME_POOLED_SUPPORT_CONFIRMATION_PREREGISTRATION.md`

Run:
`results/nonmyopic/number_game_pooled_support_confirmation/number-game-pooled-support-confirmation-20260728T170000Z`

## Verdict

**All preregistered pooled-support gates pass.**

On 32 fresh Gemini planning trees and 32 fresh GPT-5.4 target supports,
root-conditioned predictive-risk BED beats a static depth-two control that
receives the deduplicated union of every rule generated under every candidate
root and answer. The global static support contains 141--179 unique rules per
tree, versus the much smaller realized branch support.

The same fresh run also independently passes every gate from the original
powered comparison against myopic EIG, fixed-support depth two, PTS, and exact
uniform random.

## Pooled-Support Result

| Metric | Root-conditioned BED | Global static pool | Relative gain | Wins | Whole-tree 95% difference CI |
|---|---:|---:|---:|---:|---:|
| Brier | 0.18381 | 0.20452 | 10.13% | 22/32 | [-0.02992, -0.01233] |
| best-rule Hamming | 0.09231 | 0.11222 | 17.74% | 21/32 | [-0.03007, -0.01012] |

Exact-extension coverage rises by `5.69` percentage points. On target
extensions absent from the initial planning support, candidate-minus-pool
differences are `-0.02312` Brier and `-0.01829` Hamming.

The global-pool root differs from the root-conditioned root on 26/32 trees.
Pooling only the two answer supports within each root is an exact identity on
32/32 trees, as frozen: answer-consistency filtering reconstructs those
branches. The confirmed load-bearing transition is therefore which semantic
support the LLM generates after a root query, not merely how many hypotheses
are available or which answer label indexes them.

## Original Controls On The Same Fresh Trees

| Baseline | Candidate Brier | Baseline Brier | Relative gain | Brier wins | Whole-tree 95% difference CI |
|---|---:|---:|---:|---:|---:|
| myopic EIG | 0.18381 | 0.22905 | 19.75% | 31/32 | [-0.06060, -0.03300] |
| fixed-support depth two | 0.18381 | 0.23403 | 21.46% | 31/32 | [-0.06599, -0.03695] |
| exact uniform random root | 0.18381 | 0.20658 | 11.02% | 30/32 | [-0.02710, -0.01858] |
| seeded PTS | 0.18381 | 0.20512 | 10.39% | 27/32 | [-0.02849, -0.01478] |

The candidate root differs from myopic and fixed-support depth two on all
32 trees. Hamming reductions are `29.80%` versus myopic and `12.08%` versus
PTS; extension coverage and novel-target metrics also favor the candidate.

## Accounting

- Successful responses / HTTP attempts: `576 / 579`
- Explicit zero-cost provider-error retries: `3`
- Reasoning tokens / forced exits: `0 / 0`
- Cost: `$1.4939861`
- Confirmation result SHA-256:
  `5171b5872ebcd4fc978393c5e47c3fa00bb011e3a5872264b31ff1e79660b15b`
- Base result SHA-256:
  `de3acb04ad8e333bcaba7d09a794b27085143f9e15644c26924d2e61d0db3b0b`
- Pooled result SHA-256:
  `92a9a24e6a420c39b56943667389697f98e933d4e473026222ac6b81a6df7b32`
- Tree artifact SHA-256:
  `46b9acbc981872fbcd3b24095ad43f35eb9a45099a0c2370c080b4c5b9ee1f39`
- Private raw-response SHA-256:
  `d254c79e79d203318c4662377e05fea9d3b1f47e832a1cadb00817430a5b65ee`
