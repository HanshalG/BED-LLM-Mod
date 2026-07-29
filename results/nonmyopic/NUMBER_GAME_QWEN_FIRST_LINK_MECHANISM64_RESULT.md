# Number Game Qwen First-Link Mechanism-64 Result

Run: `number-game-qwen-first-link-mechanism64-20260729T071529Z`

Status: **retrospective first-link summary**. No source status changes.

## Myopic Comparison

Path-dependent depth three selects a different root from myopic EIG on 62/64
trees. On those changed-root trees:

- simulated-risk advantage versus the myopic root: `0.02327`;
- realized exact-canonical Brier advantage: `0.01476`;
- realized-advantage stratified interval: `[0.00960,0.02027]`;
- wins/losses: `50/12`;
- score-to-realized-advantage Spearman: `0.407`;
- stratified Spearman interval: `[0.160,0.611]`.

Both cohorts have 25 wins. Their changed-root Spearman correlations are
`0.239` and `0.637`, so the positive pooled first link is heterogeneous in
magnitude but not driven by a single cohort.

## Deeper Controls

Against fixed-support depth three, roots differ on 53/64 trees. Mean realized
advantage is positive (`0.00596`) with interval `[0.00105,0.01112]`, but the
score-to-realized Spearman is only `0.127`, interval
`[-0.157,0.401]`.

Against cross-fitted depth two, roots differ on 41/64 trees. Mean realized
advantage is `0.00430`, interval `[-0.00033,0.00878]`, and Spearman is
`-0.014`, interval `[-0.351,0.325]`.

The simulator is therefore calibrated enough to reject myopic roots and
directionally useful against fixed-support planning, but it is not a
universally calibrated ranking instrument among deeper planners. This
explains why non-myopic versus myopic replicates while monotonic depth does
not.

## Scope

This uses the exact per-root canonical endpoint already present in each fixed
tree. It tests the first link from simulated ranking to endpoint utility. It
does not introduce online observation or posterior-execution noise and does
not test a second link.

Model calls and cost are zero. Public `RESULT.json` SHA-256:
`800532bebb47fcaf499f4136ad2c886ce79fc2cd3f00bc9bdb12e65ca2afe821`.
