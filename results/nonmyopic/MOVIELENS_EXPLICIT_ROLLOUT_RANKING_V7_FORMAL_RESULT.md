# MovieLens Explicit Regeneration-Rollout Ranking v7 Formal Result

Date: 2026-07-24

Run: `movielens-explicit-rollout-v7-formal-20260724T114108Z`

Status: passed every preregistered ranking gate; a sequential policy test is
authorized.

## Result

The target-blind screen evaluated all 20 frozen users and prospectively enrolled the
first four with maximum immediate EIG at least `.02`: users 201, 910, 747, and 313.
The explicit scorer then enumerated all five ratings for each of four candidates,
regenerated the semantic support, and recomputed history-free downstream
likelihoods before any candidate outcome was read.

| Metric | Explicit rollout | Immediate EIG | Seeded random |
|---|---:|---:|---:|
| Mean top-1 realized NLL regret | 0.116980 | 0.194774 | 0.138142 |
| Wins versus immediate EIG | 4/4 | - | - |

The explicit score's Spearman correlation with negative realized branch NLL was
`+0.273529`, above the preregistered `+.25` threshold. Its mean regret improvement
over immediate EIG was `0.077794`, above the `.02` threshold. It beat immediate EIG
on all four users and was better than seeded random.

## Audit

The run made exactly 232 physical requests: 40 screening, 160 hypothetical
regeneration/likelihood, and 32 realized-branch requests. It used zero reasoning
tokens and cost `$0.95915395`. Raw counts were exactly `20, 20, 80, 80, 16, 16`
for initial profiles, initial likelihoods, hypothetical profiles, hypothetical
likelihoods, realized profiles, and realized likelihoods.

No held-out rating entered a prompt. Candidate outcomes were hidden from the rollout
scorer and were read only for the subsequent realized-branch evaluation. Raw model
text remains private and untracked. Its SHA-256 is
`2f97d4af00a9cbbf325a6df478c2e8549bab4aff3f801e9e611cf376487fcc1c`.

This is a positive, preregistered mechanism-ranking result at `n=4`, not yet a
powered sequential-policy claim. The next authorized step is a fresh-cohort policy
comparison using this exact explicit transition.
