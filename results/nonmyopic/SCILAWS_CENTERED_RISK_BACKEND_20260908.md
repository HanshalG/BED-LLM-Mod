# Centered risk removes the identified memory obstruction

Added explicit risk_backend='centered' to plan_batched; pairwise remains default.
Store centered, target-weighted particle means and squared norms rather than a
particle-by-particle distance matrix. Include exactly the same conditional target
noise. If moment subtraction is near cancellation, recompute that row with a direct
two-pass weighted variance, not arbitrary clipping or a changed loss.

53 focused tests pass in1.79s: full root agreement with pairwise h0/h1/h2/h3 in
adaptive/open-loop repeated mode, large target offsets, singleton and nearly
concentrated posterior weights, invalid backend, and existing default/noise/repeat
tests. Scoped lint passed before the final test addition; implementation unchanged.

At the calibrated2048 particles,8actions,64targets,16branches,h3, the actual memory
preflight estimates72.015625MiB for pairwise versus42.03125MiB for centered. Under
64MiB the former admits zero rows and the latter one. A test intercepts the first
branch-generation call: centered reaches it, pairwise rejects before it. No branch
or source outcome is generated in that test. These are conservative code workspace
estimates, not measured total resident memory or runtime certification.

The memory obstacle identified by the512-per-family Sobol calibration is therefore
addressed, not all planning feasibility. Full h3 state/time costs and unequal-noise
quantile accuracy remain untested. The myopic-policy evaluation helper still uses
the pairwise backend and distinct designs; align it before any paired deployment.
No default backend switch or scientific run is authorized by these software tests.

Next: prospectively qualify unequal-noise one-step particle integration against
an independent fixed-particle reference at the calibrated count and full action
menu, with explicit error/time/state caps and a cost preflight. Then assess complete
depth2/3 cost and decision refinement. Keep particle approximation to the continuous
model separate from integration error within the particle model. Source calibration,
useful LLM proposals and non-myopic efficacy are still independent unfinished gates.

No source/modelcalls/spend; usage220.376693994,balance24.623306006,London spend0.
No active process, automation paused, full goal unfinished.
