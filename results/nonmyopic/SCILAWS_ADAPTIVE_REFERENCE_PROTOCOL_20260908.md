# Independent adaptive residual reference

Freeze before one execution of scilaws_adaptive_reference_audit.py. Use all four
already-opened synthetic mixed-refinement histories, unchanged model constants,
both actions, repeats allowed. No source measurements or new scientific endpoint.

Use SciPy QUADPACK adaptive integration directly over the real observation line,
not the existing component-Jacobi nodes. Terminal integral is analytic expected
within-family risk plus density-weighted between-family variance, with direct
posterior means/weights calculated independently from the quadrature path.
Test against separately integrated full conditioned mixture risk.

Attempt depths1/2 with both roots, each plan sharing five seconds/100000 integrand
evaluations, epsabs=epsrel=1e-8 and100 adaptive subdivisions. At depth2 integrate
the minimum of independently integrated terminal action values minus the family
potential, adding the exact forced-root potential expectation. Record QUADPACK
outer and maximum encountered inner error estimates separately. These are not
rigorous nested-error bounds. No partial root set may count as a complete plan.

Bank every cap/nonconvergence; do not retry with larger limits. Completed h1
values compare existing orders4/8/16/64. Completed h2 values can be compared with
the banked fixed-rule audit offline. A failure means no reference qualification,
not permission to substitute the previous unverified value. A diagnostic pass
does not authorize source or LLM execution or a new estimator deployment.
