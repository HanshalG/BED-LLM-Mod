# MovieLens Receding Explicit Policy v9 Formal Result

Date: 2026-07-24

Run: `movielens-receding-policy-v9-formal-20260724T124635Z`

Status: failed the preregistered efficacy gate; close the receding explicit
policy at this sample and apparatus.

## Protocol Audit

The source invocation stopped after 256 requests because one of 160 simulated
first-round profile responses was truncated. It had not read a candidate or
held-out rating. The preregistered hash-locked recovery reused the other 255
responses and replaced only frozen index 13.

The combined run completed exactly 1,257 requests, equal to
`417 + 40 * 21`: 1,256 scientific requests plus one operational replacement
over 21 unique round-two states. It used zero reasoning tokens and cost
`$6.67587505`. Four probability rows were normalized under the frozen
`[0.90, 1.10]` tolerance. All eight users were prospectively enrolled, every
policy path used two distinct queries, and endpoint ratings remained sealed
until the corresponding trees and paths were fixed.

The failed source and completed recovery raw SHA-256 values are respectively
`bec011754b9f7c90e01f484b1754a8ba86e53dac0b1e358536474febd6909ee0`
and
`ce32f025f2dc1120dbcd93564faefa71149845897c3847e820cc9f39a46d17b1`.
Raw model text remains private and untracked.

## Result

| Policy | Mean round-1 NLL | Mean final NLL | Round-2 improvement |
|---|---:|---:|---:|
| Receding explicit | 1.5283 | 1.5094 | 0.0190 |
| Immediate EIG | 1.5225 | 1.5049 | 0.0176 |
| Seeded random | 1.5165 | 1.4942 | 0.0224 |

Receding explicit was worse than immediate EIG by `0.00447` mean final NLL
(paired SD `0.03941`, SE `0.01393`, bootstrap 95% interval for explicit
improvement `[-0.0290, 0.0217]`) and won 4/8 users, below the frozen 5/8 gate.
It was worse than random by `0.01520` mean NLL and therefore failed all three
efficacy criteria.

| User | Explicit | Immediate | Random | Explicit improvement vs immediate |
|---:|---:|---:|---:|---:|
| 880 | 1.3602 | 1.4252 | 1.3692 | +0.0649 |
| 246 | 1.6803 | 1.6990 | 1.6223 | +0.0187 |
| 59 | 1.4940 | 1.4368 | 1.4904 | -0.0572 |
| 308 | 1.4234 | 1.4253 | 1.4853 | +0.0018 |
| 465 | 1.4564 | 1.4780 | 1.4840 | +0.0216 |
| 632 | 1.6120 | 1.5882 | 1.5298 | -0.0238 |
| 497 | 1.5547 | 1.5349 | 1.5468 | -0.0198 |
| 339 | 1.4939 | 1.4518 | 1.4254 | -0.0420 |

## Diagnosis

The null is not caused by policy collapse. Explicit and immediate selected
different first queries for all eight users and different second queries for
seven; the 21 unique round-two states nearly span the maximum 24 policy states.

Across the 21 unique first-round queries actually selected by any policy, the
explicit score retained a modest rank correlation with negative realized
round-one NLL (`Spearman rho = +0.370`), while immediate EIG's correlation was
`+0.058`. That directional signal was not strong enough for top-one policy
selection: explicit's chosen first query was worse than immediate by `0.00587`
mean NLL and won only 3/8 users. Replanning then improved every arm by a similar
amount and did not recover the first-round selection variance.

The v7 one-transition ranking result therefore identified a real aggregate
signal, but its four-user all-branch regret advantage did not transfer into an
eight-user sequential top-one policy advantage under one realized rating per
query. V8 showed that composing the semantic simulator twice reverses score
fidelity; v9 shows that avoiding composition alone is insufficient. The
remaining bottleneck is reliable action-level ranking under realized outcomes,
not a lack of adaptive policy diversity.
