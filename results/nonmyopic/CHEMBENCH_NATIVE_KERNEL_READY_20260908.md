# Explicit native numerical engine

## Changes and verification

Add an optional `NativeEnvelopeGaussianModel`, selected explicitly with
`--engine native`; the default remains the NumPy envelope model. The native
engine needs the pinned packages in `scripts/requirements-chembench-native.txt`
and a working C compiler. Cython builds a small local extension in
`~/.cache/bed-native-quantiles`; there is no GPU, cluster or model service.

The compiled loops use SciPy's own normal CDF and inverse CDF. They preserve the
same full-mixture likelihood, envelope knots, Legendre nodes and masses. The
native CDF implements NumPy's contiguous pairwise summation order: an initial
sequential reduction missed the concentrated-mixture equivalence test by about
9e-10 in observations. The test was not weakened. Exact double-precision CDF
saturation beyond z=9 or below z=-40 avoids unnecessary special-function calls
without removing any mixture component. Brackets and the bisection fallback
remain in place. Dimensions are checked before unchecked native array access.

Batch construction avoids repeated Python per-belief dispatch. Precomputed
Legendre table storage is counted against the existing workspace limit. Both
real and batched simulated posterior updates now use the same max-shifted
log-sum-exp normalization, preserving log weights and the physical likelihood.

112 focused tests pass in 25.45 seconds, including compiled/NumPy branch and
complete-policy equivalence at h1/h2/h3 in adaptive and open-loop modes,
concentrated/unequal-noise/tiny-tail/zero-support cases, exact quadrature masses,
equal-mean envelope handling and malformed dimensions. Existing strict equality
between scalar conditioning and simulated branch posteriors remains passing.
Scoped lint and whitespace checks pass.

## Frozen numerical gates

- `chembench_native_refinement/20260908-v2/RESULT.json`: all unchanged one-step,
  three-step, refinement and constructed-adaptivity gates pass. The 64-branch
  three-step sufficient-statistic error is 0.00001358496; runtime is 3.586s.
- `chembench_envelope_refinement/20260908-v5/RESULT.json`: the default NumPy
  engine also passes after shared normalization/refactoring.
- Native preflight explicitly requires the native V2 source hashes and engine
  identity. NumPy preflight binds V5. A missing extension/dependency or changed
  binding fails closed; there is no silent fallback during a native run.

Two earlier exploratory native profiles still exceeded 60 seconds. They predate
the final batched-rule/normalization combination and remain banked under
`chembench_public_kernel_profile/20260908-native-v1` and `...-v2`. They are
diagnostic timings, not complete source-policy results; intermediate development
kernel versions are identified by hashes but are not independently packaged
runtime snapshots. Do not treat them as replay-qualified experimental endpoints.
Host load varied, so timings are not a controlled hardware speedup study.

## Next authorized execution

After this code and its qualifying artifacts are pushed, run one new-version
complete source pilot with `--engine native`. The frozen source physics remains
SHA `8e1fc9df41d177fa80b2e500c22c663e3a78a980ecedaa355c667116cc2e1d36`:
16 prior particles, 64 branches, 8 independently sampled worlds, all six
deployable controls and a separate population oracle, unchanged seeds, loss,
noise and 60-second/decision, 300-second/world, 2400-second/panel caps.

All deployable initial plans must complete before the hidden-world constructor
opens. An execution failure is not a scientific null. No automatic retry,
support reduction, shallower fallback or relaxed cap is permitted. Complete
source results still need independent replay and paired analysis before any
downstream LLM semantic gate. This checkpoint makes no source efficacy or
monotonic-depth claim and authorizes no paid calls. Cost so far: $0.
