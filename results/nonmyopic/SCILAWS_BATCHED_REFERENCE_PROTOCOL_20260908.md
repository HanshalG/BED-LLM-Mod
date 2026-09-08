# Joint adaptive action integral workload

Integrate all8 scalar-Gaussian action distributions together by mapping each
action's full +/-8-sigma union domain affinely onto[0,1]. Include its Jacobian.
One quad_vec refinement partition evaluates all8 posteriors per callback and
charges8 evaluations. Stable max-shift normalization, unchanged centered target
risk and full particle support. Different actions need not share physical y.

Start with one adaptive interval, not sixteen forced initial subdivisions.
This is an explicit new integration algorithm; previous scalar references stay
immutable. Retain maximum-norm quadrature error+tail<=1e-7 and per-action mass
error<=1e-8. A narrow separated-mode test must fail closed if unsampled; no claim
that a numerical error estimate is a universal certificate. Limit cache to2MiB,
conservative workspace64MiB, shared5s/100000 charged evaluations for the menu.

Test first task's three histories, seed1304, action0 child15, unchanged2048
particles and8 actions against the24 banked independent scalar values. Exactly
one new all-action integral per case, no old reference or planner rerun.
Workload criterion in allthree: discrepancy<=1e-7, time<=0.04s, evaluations<=800.
The cost targets follow from an optimistic120 child menus (15 children x8 roots):
4.8s and96000evaluations, before overhead. They are necessary engineering targets,
not guaranteed sufficient whole-decision budgets or measured full-tree costs.

Record accuracy and each cost criterion separately. No repeat or post-result
mesh tuning, no new depth grid unless useful throughput is demonstrated. A pass
still requires full fixture/continuation qualification and actual whole-decision
outer accuracy and runtime. No scientific gate changed, source/model calls0,
daily spend0, automation stays paused, full goal unfinished.
