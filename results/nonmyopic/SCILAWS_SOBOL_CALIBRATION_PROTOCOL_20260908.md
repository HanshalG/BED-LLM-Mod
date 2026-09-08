# Scrambled Sobol joint posterior screen

Prospective alternative to IID, not a change to posterior physics or gates.
Per family use one independently seeded scrambled52-bit Sobol net with dimension
1+coefficient_dimension and power-of-two count. First coordinate transforms through
Gamma(shape,1) inverse CDF to variance; remaining coordinates through normal inverse
CDF and the same conditional Cholesky solve. Endpoints fail, never clip tails.
Family posterior mass is unchanged; no new conditioning or point-estimate noise.

Repeat the exact full IID screen: all24fixtures, counts32/128/512, seeds1304/1305,
the same three hypothetical updates and all mean/variance/density/family/ESS gates.
144draws,576comparisons; each count must pass every comparison in both streams.
Report all failures and the same separate h3 workspace check. No seed/case exclusion,
IID rerun, cap change or source/model calls. A calibration pass is not1e-4 decision
accuracy, h3 feasibility or useful LLM-generated model evidence.
