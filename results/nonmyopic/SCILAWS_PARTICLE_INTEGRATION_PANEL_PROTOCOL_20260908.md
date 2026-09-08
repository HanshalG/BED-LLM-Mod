# Full calibrated-particle integration panel

Prospective zero-call continuation of the pinned three-case workload, not a
source or LLM experiment. Geometry SHA256
5e9f2bd902fa9de251cbe033bdd7dd4d5ad8fc79d870e80add7920ed92a18197.
Reuse exactly the three first-task/seed1304 cases in workload artifact
4597f7e59916cd633fcd0bddb0c7eac96d1b9239e1c6c55aa2b8dc6722ba1b57;
do not rerun them. Run the other 45 cases once, with exclusive per-task/seed
artifacts and no overwrite or retry. All eight ordered public tasks, zero/affine/
quadratic software histories and seeds1304/1305 are required.

Use unchanged 512 joint Sobol posterior particles per family and exact family
masses, particle-specific noise, noisy-target loss, eight actions and 64 targets.
Compare only 32/64 quantile branches at horizon one, centered risk backend.
Each plan retains 5 seconds, 100000 states and 64 MiB workspace. Each case's
independent full-mixture adaptive reference shares 5 seconds/100000 integrand
evaluations across all eight actions. Root error and reference action regret
must each be <=1e-4; every reference error estimate plus tail bound <=1e-7,
mass error <=1e-8, all finite. Quadrature estimates are not mathematical error
certificates. Each branch count qualifies only with all 48 cases passing.

Test actual horizon-three memory admission at both branch counts by stopping
before any branch generation, without a depth experiment or cap relaxation.
Passing horizon one does not qualify deep integration, full episodes, source
calibration, useful LLM proposals, or scientific efficacy. No source readings,
paid calls, holdouts, or deployment authorized. Preserve every null/prefix.
