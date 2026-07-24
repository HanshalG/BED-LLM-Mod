# MovieLens Depth-2 Semantic Policy v8 Formal Result

Date: 2026-07-24

Source run: `movielens-depth2-policy-v8-formal-20260724T120636Z`

Recovery: `movielens-depth2-policy-v8-recovery-20260724T121944Z`

Status: failed the preregistered policy gate; close this entropy planner.

## Result

The 48-user screen prospectively enrolled users 455, 933, 395, and 606. All three
policies traversed the same fully precomputed semantic transition tree.

| Policy | Mean final held-out NLL |
|---|---:|
| Depth-2 semantic rollout | 1.515465 |
| Depth-1 explicit rollout | 1.483811 |
| Immediate EIG | 1.514733 |

Depth 2's mean improvement was `-0.031654` versus depth 1 and `-0.000732` versus
immediate EIG. It beat depth 1 on 3/4 users but immediate EIG on only 1/4.

| User | Depth 2 | Depth 1 | Immediate EIG |
|---:|---:|---:|---:|
| 455 | 1.438615 | 1.452322 | 1.428541 |
| 933 | 1.759516 | 1.546312 | 1.695137 |
| 395 | 1.455993 | 1.457349 | 1.455993 |
| 606 | 1.407737 | 1.479260 | 1.479260 |

User 933's `+0.213204` loss versus depth 1 reverses three smaller depth-2 wins.
The depth-2 planner predicted its selected first query to have the lowest expected
terminal entropy for that user, yet its realized path had the worst NLL. This is
consistent with predictive entropy rewarding a confident but wrong terminal belief.

## Recovery And Audit

The paid run made exactly 1,056 requests, used zero reasoning tokens, and cost
`$5.00894919`. It failed before outcomes because two of 25,600 terminal rows summed
to `.90` and `.94`. The preregistered local recovery required the frozen raw hash,
normalized only those two finite nonnegative rows under a `[.90,1.10]` tolerance,
and made zero endpoint calls. Every efficacy gate remained unchanged.

Raw model text remains private and untracked. The source and replay raw SHA-256 is
`11f1311024b559008db543d037b518f5e89a35b2d5b800c1b8ce7494ffc87cfb`.

V7 established that explicit one-step semantic transitions can improve query
ranking. V8 does not show that extending this entropy objective to depth 2 improves
sequential decisions. A larger confirmation is not authorized for this planner.
