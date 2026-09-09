# Known-structure parameter-integration qualification

Before computing particle comparisons, freeze one numerical fixture: rate=k*C_A,
log-uniform k in[.01,100], true fixture coefficient1.3; six inputs and fixed
standardized residual offsets in parameter_integration_reference_audit.py.
Gaussian log1p-rate noise sigma.05. Four target inputs assess posterior prediction,
not a hidden scientific endpoint. All other chemistry inputs fixed.

Independent log-parameter Gauss-Legendre orders1024/2048 must agree to absolute
1e-9 in posterior predictive means, variances and log evidence. Compare existing
ExecutableBeliefPool prior-draw importance weighting at32/256/2048particles for
every seed0..7. Do not select seeds, enlarge counts after outcomes, change the
prior or tune observations. Same public full history for all.

Numerical qualification per replicate requires max predictive mean error<=.01,
absolute log-evidence error<=.1, ESS>=10. Report all metrics and all failures.
These are mechanics tolerances, not a new BED or model-proposal efficacy gate.
A pass in this one-dimensional case cannot establish high-dimensional accuracy.
No LLM call, model-generated structure, closed cohort or benchmark outcome used.
This audit authorizes no paid run; preserve the old fitter and bank results once.
