# Analytic terminal-risk interval replay

For fixed continuous mixture, lower bound is known-family posterior risk. Upper
bound is risk of the feasible LMMSE target predictor: prior weighted risk minus
weighted squared target/observation covariance divided by observation variance.
Include within-parameter and between-family covariance, independent future target
noise when requested. Pad floating endpoints by1e-12 times max(1,prior risk).
These are working-model bounds, not source-misspecification guarantees.

Replay every completed terminal call from the hash-bound existing one-root work
trace; generate no new integrals or source measurements. Test interval containment
against banked adaptive values allowing their reported numerical error. Count
calls for which half width<=1e-8, the existing absolute inner tolerance, and sum
their recorded work as potential savings. Do not loosen the threshold if none
are eligible. A positive width test alone does not implement or authorize a
shortcut; first verify the bound independently against full linear-rule loss.
