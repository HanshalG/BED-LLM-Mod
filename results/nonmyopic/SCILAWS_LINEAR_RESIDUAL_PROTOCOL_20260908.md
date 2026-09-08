# Analytic-baseline residual interpolation

New explicit adaptive representation, no change to failed full-value candidate.
Use exact continuous-model linear-prediction risk B_a(y), fit
(B_a(y)-V_a(y))/(1+z^2), reconstruct B_a(y)-(1+z^2)*fit before the action minimum.
Signed residual estimates are permitted without clipping, since numerical inner
errors can yield tiny signed differences; reconstructed risks must be finite
and nonnegative. Algebraically, normalized reconstructed action-value error is
exactly minus residual approximation error at every fresh check point.

Keep all four histories, both roots, one shared5s/100000 budget;17 initial nodes,
65-node/eight-pass caps, disjoint golden-ratio final checks,2e-5 normalized
threshold,1e-7 inner error check,1e-6 tail envelope and existing integration rule.
Freeze before one diagnostic, retain partial/failing roots without qualification.
No source/LLM calls or deployment authorization; sampled checks are not uniform
error proofs. Compare any completed candidate with independently saved values.
