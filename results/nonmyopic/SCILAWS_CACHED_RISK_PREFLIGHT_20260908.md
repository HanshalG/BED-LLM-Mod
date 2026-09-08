# Cached precision and target-risk preflight

Previous turn completed the reference prior and exposed a runtime limit. This
turn profiled the same public baseball prior under the same five-second cap.
The profile recorded 34276 linear solves, 1.551s direct solve time / 1.864s
cumulative, out of 4.906 profiled seconds. Full target moments and repeated
conditioning were prominent callers. No source observations were involved.

## Change and verification

Cache exact precision/feature solves in a bounded 1024-entry cache. Precision
depends on actions, not observation values, so branches reuse the same solve.
Results are read-only. There is no approximate inverse, support reduction or
likelihood change. Cache sizes are bounded, not disk-backed transition archives.

For scalar target risk, precompute X' W X and use
E[sigma2] trace(Lambda^{-1} X' W X) for within-component integrated variance.
Retain the original centred between-component prediction variance and optional
observation noise. Per-model target leverage caches have 256 entries. Full
target moments remain available as an independent comparison path.

89 SciLaws tests pass in4.35s; scoped lint passes. New tests compare scalar risk
with full target moments on four conditioned histories for all eight development
geometries, and verify cached solves/read-only bounds. An additional complete
eight-task h1 root-vector comparison against the full-moment risk implementation
passes: maximum absolute action-value difference 2.7755575615628914e-17, with
identical selected roots. This is not h2/h3 quadrature qualification.

## Bounded execution

Implementation pushed at b6f85c08 before the changed-version preflight. Result:
SCILAWS_REFERENCE_PREFLIGHT_CACHED_20260908.json, SHA256
202a154fbedd6f59854deab338b06fbe1ceed86066e7823fbc536c3463ea2953.

- All8 h1 plans complete; all8 h2 plans hit the unchanged five-second cap.
- h3 is skipped after each h2 cap, as specified.
- The first5 h1 timings are .085-.166s, versus .244-.656s in the earlier run.
- A separate root-equivalence check overlapped the latter part of this preflight.
  The final3 timings are therefore potentially contended; do not report a clean
  full-panel speedup ratio or infer their isolated runtime from these readings.
- The first5 h2 caps preceded that overlap. Caching is insufficient to meet the
  limit on those tasks; this is not merely a conclusion from contended timings.
- Both processes exited. No source measurements, paid calls or efficacy endpoints.

The next meaningful implementation change is batched component conditioning and
risk evaluation over whole sets of hypothetical observations, preserving raw
likelihoods, full support and the same quadrature rule. Profile it and compare
complete action values before a new isolated runtime check. Do not repeat this
unchanged version, relax caps or launch a source pilot to evade the dependency.

The four-feature-family reference still does not establish an LLM role. Source
calibration, the full paired policy protocol, licensing review, useful executable
LLM proposals and anticipated-discovery evidence remain unfinished. Account and
London ledger are unchanged at zero daily spend; automation remains paused.
