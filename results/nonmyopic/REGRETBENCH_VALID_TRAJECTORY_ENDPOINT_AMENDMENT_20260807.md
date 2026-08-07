# RegretBench Valid-Trajectory Endpoint Amendment

Date: 2026-08-07

Status: frozen before any RegretBench support or policy model response.

## Problem

The original endpoint scored truth mass from the regenerated support even when
an executed question was unsupported or the second question repeated the first
official facet. Those observations are deterministic and truth-independent in
this environment. A support redraw could nevertheless improve after such a
dialogue, incorrectly crediting internal re-prompting as experimental
information.

## Amendment

A valid two-action trajectory requires both official mappings to be supported
and the second mapped facet to differ from the first. For every policy:

- a valid trajectory retains the preregistered exact generated-likelihood
  primary endpoint and fresh-regeneration secondary endpoint;
- an invalid trajectory receives primary truth mass `0`, Brier `1`, and log
  loss at the frozen `1e-12` probability floor;
- its scored fresh-regeneration truth mass is also `0`, while the raw redraw
  mass is retained under an explicitly descriptive field; and
- first-step scored truth mass is `0` when the first action is unsupported,
  with raw first-step mass retained descriptively.

The same validity penalty applies to the optional Luna fresh-regeneration
baseline, which remains unable to affect primary status. Public rows record the
validity flag and raw/scored masses. The independent verifier reconstructs all
fields from raw responses and official mappings.

No prompt, task, seed, request count, policy selection, bootstrap threshold, or
budget changes. This amendment can only preserve or worsen an invalid path's
score; it cannot rescue a result.
