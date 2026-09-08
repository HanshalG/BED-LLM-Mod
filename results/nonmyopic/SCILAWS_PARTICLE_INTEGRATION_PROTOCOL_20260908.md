# Calibrated-count integration workload

Prospective first-task software diagnostic: all3 initial scenarios, full8actions,
Sobol512/family using calibration seed1304 and identical SeedSequence construction.
Reference integrates the full finite Gaussian-mixture density once, not each particle
separately. Domain spans min(mu-8sigma) to max(mu+8sigma), enlarged only to ensure
global risk bound times Gaussian tail mass<=1e-9. Seed16 uniform integration segments,
adaptive vector quadrature jointly checks mass and normalized risk. All particles
enter every density/posterior evaluation; no support filtering.

Independent reference max reported risk error<=1e-7 and mass discrepancy<=1e-8;
analytic omitted risk tail included. Mass agreement is not proof against every
possible unresolved narrow feature; adaptive errors remain estimates. One shared
5second/100000integrand-call budget across all8actions in a scenario, no peraction
reset. Compare centered-risk quantile plans with4/8/16/32/64branches, each with
5second/100000state/64MiB caps. Require all root errors and action regret<=1e-4,
with reference error+tail<=1e-7. Preserve every candidate failure and partial reference.

Cost preflight: h1 candidate at most512leaf beliefs at64branches, within statecap;
workspace checked by actual planner. Reference integrand cost scales linearly with
particles*targets but adaptive call count is unknown and bounded explicitly. This
small full-action workload precedes any full48-case integration grid. No source,
h2/h3 or LLM permission. Particle-vs-continuous calibration is a separate prior gate.
