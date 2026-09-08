# Interval-guided order4 terminal reference

Freeze all24 fixtures/eight roots/full branch and continuation action menus.
Single-thread BLAS. Initialization before timer, all branch construction, analytic
bounds and numerical refinement within one5second/100000evaluation budget percase.
Count analytic action-bound calls as well as adaptive integrand evaluations.

Refine the branch with largest weighted uncertainty, then its unrefined action
with smallest lower bound, provided that lower bound<=current minimum upper bound.
Ties remain eligible. Intersect analytic bounds with numerical value +/- reported
error (required<=1e-7); conflicts fail, never widen bounds or suppress actions.
Stop at5e-5 root midpoint terminal uncertainty. Exhausted budget fails closed.

All192 roots intended, ordered prefix retained on failure, no retries. Full24-case
completion is required for operational coverage. These are numerical estimated
enclosures, not rigorous bounds after adaptive errors enter. Outer-order4 error
is explicitly unbounded; this cannot qualify continuous h2 or h3/source/LLM use.
