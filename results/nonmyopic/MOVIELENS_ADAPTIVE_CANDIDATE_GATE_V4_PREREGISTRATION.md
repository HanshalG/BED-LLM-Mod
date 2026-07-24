# MovieLens Adaptive-Candidate Profile Gate v4

Date: 2026-07-24

Status: preregistered mechanism gate. No v4 response or endpoint has been viewed.

## Motivation

V2 showed a genuine but concentrated load-bearing profile transition. Its fixed four
movies were weakly discriminative for 7/12 users. V3 tried to manufacture profile
contrast and failed serving stability. V4 instead applies adaptive design to the legal
query set while restoring the successful unconstrained v2 semantic representation.

For each user, the likelihood model scores 16 movies the user rated, selected without
using rating values. The four highest current-support EIG movies become the branch
set. This tests whether querying where the natural semantic belief actually disagrees
creates a robust path-dependent opportunity.

## Frozen Fresh Population And Candidate Construction

Users must rate the same four public-history movies and at least 24 other movies.
Every v1-v3 user is excluded. Of 79 fresh eligible users, seed `24305` freezes:

- smoke-only users: `113, 130`;
- formal users: `158, 194, 227, 234, 323, 468, 494, 551, 579, 679, 710, 854`.

For each user, the 16 rated non-history movies with greatest global rating-presence
count form the candidate pool, ties broken by movie ID. This uses presence only, never
rating values. Seed `24305000 + user_id` selects eight held-out movies from the
remaining rated movies. The likelihood model scores the 16 candidates and eight
held-out movies under the six initial profiles. The top four EIG candidates, ties
broken by movie ID, are frozen as that user's branches before any candidate rating is
read.

## Models And Gates

V2 mechanics are unchanged:

- non-thinking Gemma 4 26B A4B generates six broad semantic profiles;
- after a realized rating it generates six non-copy replacements with explicit
  evidence effects;
- the branch retains the two most compatible old profiles;
- non-reasoning GPT-5.4 Mini receives profiles and movie metadata only, never history;
- all outcomes and endpoints are recorded ratings;
- raw responses remain ignored/private and committed artifacts are hash/metric-only.

The smoke remains exactly 10 requests over two disjoint users and one top-EIG branch
each. It must pass all v2 schema, privacy, history-isolation, and replay mechanics.

The formal run remains exactly 120 requests over 12 users and four branches. Every v2
numerical gate is unchanged: mean oracle NLL improvement `>=0.05`; improvement count
`>=6/12`; branch-spread count `>=6/12`; mean immediate-EIG regret `>=0.03`; regret
count `>=4/12`; mean maximum EIG `>=0.02`; and maximum-EIG count `>=8/12`.

Passage authorizes only a fresh target-blind ranking-fidelity gate. Failure closes this
exact adaptive-candidate apparatus before policy/depth. No threshold tuning, pool-size
change, post-hoc subset, alternate split, or history bypass is allowed after responses.

OpenRouter ledger ceiling `$70.38480269545715`; run cap `$0.90`; projected formal
reservation `$0.60`; concurrency `64`. Check live and ledger balances before each
paid stage and use the lower remainder.
