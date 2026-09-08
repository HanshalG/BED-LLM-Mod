# Switch-partition diagnostic freeze

One opened empty-history case, root0, depth2. No full panel. Generic17-point
scan over predictive standardized coordinates[-8,8], all pairwise action gaps,
Brent roots on sign-changing brackets (xtol1e-8,maxiter32), exact-zero scan
points also retained. No use of the preceding observed numerical switch values.
Retain bounded256-entry exact observation cache only within a root computation.

Split the complete real line at detected points, apportion absolute1e-8
tolerance equally over segments, retain relative1e-8 and100 subdivisions each.
Evaluate the actual minimum inside every segment, not a fixed assumed action.
Unseen roots/tails are not removed. Every actual inner integrand and every newly
solved scan/root/outer observation counts against100000; five-second cap remains.

Tests cover full-domain affine Jacobian and analytic expectation of the minimum
of two shifted quadratics under Student-t6. A completed single root is still
not a full-plan reference or source qualification. Bank cap/failure once; no
larger limits, observed-root hardcoding or automatic panel follows a failure.
