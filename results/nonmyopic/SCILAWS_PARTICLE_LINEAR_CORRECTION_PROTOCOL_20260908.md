# Particle linear-predictor correction workload

Prospective numerical candidate only. No source or model calls. First public
task, all three unchanged histories, seed1304, 512 Sobol particles per family.
Reuse references from artifact4597f7e59916cd633fcd0bddb0c7eac96d1b9239e1c6c55aa2b8dc6722ba1b57.
Geometry remains5e9f2bd902fa9de251cbe033bdd7dd4d5ad8fc79d870e80add7920ed92a18197.

For centered weighted targets F and scalar observation Y, use:

    R_linear = R_prior - ||Cov(F,Y)||^2 / Var(Y)
    R_Bayes = R_linear - E ||E[F|Y] - (E[F]+Cov(F,Y)/Var(Y)*(Y-E[Y]))||^2.

The identity follows from conditional-mean orthogonality and includes independent
target noise. It retains the full finite particle posterior and unequal Gaussian
measurement variances. Only the final nonnegative residual expectation uses
quantile quadrature. Unlike bounded posterior risk, this residual can grow in
the tails: reduced quadrature error is a hypothesis, not a guarantee. Independent
infinite-domain integration tests verify the identity at1e-8; a128-node smoke
showed2.74e-5 error, so the original1e-5 smoke assertion was not met. The scientific
1e-4 gate below is unchanged from the previous panels, not inferred from this smoke.

Run4/8/16/32 nodes once on all three cases and all8 actions. Each count/case shares
5 seconds,100000states and64MiB. Every root error and chosen-action regret must
be <=1e-4 versus banked independent references. No best-case selection, clipping,
tail suppression, posterior change or repeated reference computation. Count
qualifies for a full-panel proposal only if all three cases pass; no full-panel
or deep qualification follows directly. If no low count passes, bank the null
and do not iterate more counts on this workload. Store all candidate prefixes.
