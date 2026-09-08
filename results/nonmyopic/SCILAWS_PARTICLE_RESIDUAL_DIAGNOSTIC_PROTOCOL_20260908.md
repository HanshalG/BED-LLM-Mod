# Residual diagnostic, not a new policy qualification

Retrospectively investigate both correction failures (task2/seed1305 and
task6/seed1304 affine) plus the original first-task/seed1304 affine control.
Selection is failure-informed and cannot support generalization or efficacy.
Same geometry, posterior draws, targets, noise and full8-action menus.
Reuse saved raw-risk and32-node correction results; no new count or correction
candidate, no source or inference calls. Regenerate32 quantile locations only
to define the outside-node region; charge these32 belief evaluations per action.

Independently integrate full Gaussian-mixture density times squared linear
prediction residual. Split the integration at both extreme32-node quantiles,
and record contribution outside them. Domain contains mu_i +/-r sigma_i for
every particle, r>=8. For M=max_i||F_i-EF|| and slope b, use

    residual^2 <= 2M^2 + 2||b||^2(Y-EY)^2
    tail bound <= 2M^2 P(|Z|>r)
      + 4||b||^2 sum_i w_i[(mu_i-EY)^2 P(|Z|>r)
                           + sigma_i^2 E(Z^2 1{|Z|>r})].

Increase r only until this analytic bound <=1e-9. Adaptive quadrature error
estimate plus bound must be<=1e-7, mass error<=1e-8, finite values required.
Each case shares5s/100000 charged evaluations across8 actions; preserve prefixes
on failure. Fixed sixteen initial domain subdivisions plus extreme-node splits.
Quadrature error estimates remain numerical evidence, not rigorous certificates.

Compare reconstructed risk with the banked independent risk, and residual
expectation deficit with saved correction error. Nonzero outside-node contribution
does not imply those probability weights were literally dropped by quantile
quadrature: nodes represent intervals. Report both the contribution and net
deficit, not a simplistic missing-mass claim. A consistent diagnosis may motivate
explicit tail integration, but cannot rescue the closed correction panel.
