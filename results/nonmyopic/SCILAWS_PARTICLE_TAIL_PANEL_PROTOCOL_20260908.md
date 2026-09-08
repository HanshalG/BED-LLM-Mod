# Explicit-tail residual integration panel

New numerical candidate informed by the banked residual diagnosis, not a rerun
or rescue of the failed ordinary-quantile rule. Freeze probability boundaries:

    [0, 1e-5, 1e-3, .05, .5, .95, .999, .99999, 1]

Use Gauss-Legendre order2 or4 within every interval, for16 or32 TOTAL branches.
All interval masses retained, full Gaussian-mixture quantile inversion and raw
likelihood conditioning unchanged. The linear-predictor residual is the only
integrated quantity. No posterior, target, noise, action or source changes.
Do not move boundaries or add counts after responses.

Evaluate all48 public task/history/seed cases once:8 tasks, zero/affine/quadratic,
seeds1304/1305,512 joint Sobol particles per family. Reuse independent references
SHA25606f7f6c21ceda8a34ee5cc96f7199d3da677179db241d9c6b1f20e5288e0353d.
All96 new candidate plans include eight actions,64 fixed targets and noisy-target
loss. Each count/case shares5s,100000states,64MiB. Full-root absolute error and
reference action regret <=1e-4; all48 cases must pass. Preserve exclusive shards
and failure prefixes. No fresh independent reference calls or banked plan reruns.

Software tests verify interval mass/polynomial moments and scalar/batch full
posterior updates, including separated unequal-noise components. They do not
establish general accuracy: the frozen full panel is the next quantitative gate.
Future-history and full-horizon accuracy remain required after a one-step pass.
Total branching must be charged at every future level; sixteen nodes are not
two nodes just because the per-interval polynomial order is two.

No source measurements, LLM calls, endpoint opening or deployment authorized.
No old gates or scientific nulls changed. Account daily limit unchanged;
automation stays paused. This is not an LLM or non-myopic efficacy experiment.
