# NewtonBench observation contract: verified incompatibilities

Date: 2026-09-08. Read-only source diagnostics, zero model calls and $0 cost.
Reproducer committed/pushed at `3e285d79` before executing the deterministic checks.
Result: `NEWTONBENCH_OBSERVATION_CONTRACT_20260908.json`.

## Findings

1. **Likelihood mismatch.** The pinned noise helper adds Gaussian noise to the raw
   signed output with standard deviation
   `max(abs(true_output * noise_level), absolute_noise_floor)`.
   At zero noise level it returns the exact output, bypassing the floor. Both scalar
   and array branches were checked using controlled noise functions, with no random
   draws. Our chemistry adapter's fixed-sigma log1p-rate likelihood is not this law.
   A faithful likelihood would generally require particle-dependent noise scales.

2. **Optics unit mismatch.** The AutoSciLab wrapper's incidence-angle bound is
   `[0.01, 1.5]`, annotated as radians. The vanilla source path wraps in degrees and
   the laws convert using `math.radians`; its public source description also says
   degrees. Under equal refractive indices, easy v1 at numeric input 1 returns
   1 degree, not 57.2958 degrees. Passing the wrapper bounds unchanged explores a
   narrow degree range. This is a contract mismatch, not evidence about planning.

3. **Nonfinite outcomes inside wrapper bounds.** Optics medium v0 at `(3,1,1)`
   returns NaN. Oscillator easy v0 at `(k,m,b)=(1,0.1,5)` also returns NaN.
   These points lie within the wrapper's declared input box. A likelihood over
   only finite real observations is incomplete there. One cannot silently drop
   such outcomes or target points and still claim fixed-target prediction.

4. **Signed outputs.** Oscillator easy v2 at the same point returns `-615`.
   The existing nonnegative-rate/log1p interface cannot represent this value.
   Taking absolute values, as the source evaluator does, changes the target.

These deterministic examples execute only selected pinned source functions. The
general wrapper, evaluator, model adapters and judges were never imported or run.
Three extraction tests passed; the source-contract assertions completed cleanly.
All five inspected source-file SHA256 values are retained in the result.

## Decision and scope

Do not spend on NewtonBench through the existing chemistry adapter or treat it as
a trivial source replacement. The raw source may still be useful, but would need
a prospectively specified observation contract, units, invalid-event semantics,
signed loss, source-faithful likelihood, and separately verified planning headroom.
We have not measured whole-benchmark validity, a failure frequency, calibration,
or h1/h2/h3 opportunity. This does not close every possible NewtonBench study.

Do not use these examples to choose an outcome-favourable domain subset, clamp
NaNs, reinterpret signed results as rates, or reuse the chemistry loss under a
new benchmark name. A physically revised environment is a new task, not a repair
that can inherit historical results or authorization.

The result narrows the next action: an environment must supply a coherent joint
observation law before either likelihood fitting or planner rankings can be trusted.
Our implemented structure interface is not yet a general physical-law interface;
expanding it without an adopted task would add more machinery without resolving
the scientific dependency. Keep closed endpoints closed and paid calls disabled.

The previous goal turn made concrete progress (sham-feedback control). This turn
adds executed source evidence that rules out a silent likelihood/unit transplant.
The full proposal-quality, sequential-policy and fresh-confirmation plan remains
incomplete. Automation stays paused; no experiment process remains running.
