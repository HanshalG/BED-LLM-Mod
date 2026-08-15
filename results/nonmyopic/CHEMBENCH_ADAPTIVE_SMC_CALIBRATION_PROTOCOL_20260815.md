# ChemBench Adaptive-SMC Calibration Protocol

Date frozen: 2026-08-15 (Europe/London)

## Purpose

Test the numerical parameter-posterior dependency identified by the continuous
ChemBench null before integrating any posterior into a non-myopic planner or
making another LLM call.

This is a correct-structure calibration. It deliberately removes structural
model selection, proposal generation, and experiment planning so that a pass or
failure can be attributed to parameter inference.

## Source Boundary

- Official `scientific-discovery/LLM-AutoSciLab` source commit:
  `acf160eb6c96897748dd92b152703b59b74efc05`.
- Active 57 ChemBench domains and released rate functions remain unchanged.
- Calibration covers the 48 compound/outside domains on easy, medium, and hard.
- Parameter-prior construction may read only v0-v3.
- The opened v4 values may be used only as the separately labelled stress
  cohort and never to construct a prior, proposal scale, assay, or threshold.
- No network, API, or model call is permitted.

## Parameter Prior

For each `(difficulty, structure)`:

1. Require identical nonempty parameter keys across v0-v3.
2. Use identity coordinates for `alpha`, `beta`, `n`, `n_inh`, `n_met`, `pKa`,
   `pKa1`, and `pKa2`; use log coordinates for all other positive parameters.
3. For each coordinate, expand the observed v0-v3 range on both sides by 1.5
   times the larger of its observed span and the frozen minimum span:
   `0.5` in identity coordinates and `log(2)` in log coordinates.
4. The prior is uniform in this transformed box, conditional on
   `pKa1 < pKa2` when both are present.
5. All draws are seeded and truth-independent.

## Cohorts

### Source-only in-prior cohort

Draw one hidden parameter vector from the declared transformed prior for every
compound structure and difficulty using seed base `2026083100`. These truths
are generated after the protocol is frozen but use no benchmark endpoint.

### Opened-v4 stress cohort

Use the already-open v4 hidden parameters for the same structures and tiers.
This cohort measures transfer to the prior edge or outside it. It cannot alter
the source-only gate and does not open a new scientific endpoint.

## Fixed History

Use the same eight assays for every truth and inference method, in this order:

1. `C_A=0.02` (frozen assay index 1)
2. `C_A=100` (index 4)
3. `C_I=50,C_A=0.1` (index 5)
4. `C_I=50,C_A=100` (index 8)
5. `C_B=0.01` (index 9)
6. `C_B=100` (index 11)
7. `C_P=20` (index 13)
8. `T=368` (index 15)

This covers substrate saturation, inhibitor interaction at low and high
substrate, second-substrate dependence, product inhibition, and temperature.
It was selected from mechanism vocabulary before any calibration response was
generated. pH is excluded because it is a shared released secondary effect,
not one of the unknown structural mechanism factors.

For truth mean `mu`, draw one common standard normal per truth/assay using seed
base `2026083200` and observe:

```text
y = max(0, mu * (1 + 0.01 * epsilon)).
```

All inference methods receive the exact same observations. Their likelihood is
the planner's frozen transformed-rate approximation:

```text
log1p(y) ~ Normal(log1p(mu), max(0.01*mu/(1+mu), 1e-4)).
```

## Query Evaluation

Use 512 fixed held-out query designs per tier, generated from the released
seven-dimensional bounds using:

- easy: `2026081701`
- medium: `2026081702`
- hard: `2026081703`

The estimator is the posterior mean of `log1p(rate)` at each query. The primary
loss is mean squared error against the noise-free truth query vector.

## Compared Methods

### Static importance sampling, 16 particles

- Draw 16 particles from the transformed prior with seed base `2026083300`.
- Compute normalized full-history likelihood weights.
- No resampling or rejuvenation.

### Static importance sampling, 100 particles

- Draw 100 particles from the same prior with seed base `2026083301`.
- Compute normalized full-history likelihood weights.
- No resampling or rejuvenation.

### Adaptive-tempered SMC, 100 particles

Run two independent banks with seed bases `2026083301` and `2026083302`:

- 100 prior particles;
- adaptive tempering to target ESS fraction `0.6`;
- systematic resampling after every tempering increment;
- three bounded random-walk Metropolis moves per rung;
- proposal standard deviation equal to half the current transformed particle
  standard deviation, with a small prior-width floor;
- maximum 80 tempering rungs;
- exact incremental marginal-evidence accumulation.

The first SMC bank shares its initial prior seed with the 100-particle static
baseline. The two SMC banks are averaged only for reported predictive loss;
each bank must independently satisfy finite and diagnostic requirements.

## Frozen Gates

The calibration passes only if all conditions hold.

### Correctness and health

1. Every posterior is finite, normalized, and reproducible from its seed.
2. Every SMC run reaches temperature 1 within 80 rungs.
3. Aggregate Metropolis acceptance lies in `[0.05, 0.90]` on every tier.
4. At least 95% of SMC runs use a nonzero rejuvenation acceptance.
5. No v4 value or query truth influences prior bounds or particles.

### Source-only predictive calibration

6. Mean two-bank SMC MSE is at least 10% lower than 16-particle importance
   sampling in aggregate.
7. Mean two-bank SMC MSE is no more than 2% worse than 100-particle importance
   sampling in aggregate.
8. SMC is strictly better than the 16-particle baseline and no more than 5%
   worse than the 100-particle baseline on each difficulty tier.
9. Practical paired wins exceed losses against 16 particles at tolerance
   `1e-8`.

### Stability

10. Across source-only truth histories, the two SMC banks' log-evidence values
    have Spearman correlation at least `0.95` on every tier.
11. Their median absolute log-evidence difference is at most `1.0` nat on every
    tier.

### Opened-v4 stress

12. Aggregate SMC MSE is strictly lower than 16-particle importance sampling.
13. The report must disclose, per tier, the fraction of v4 truth coordinates
    outside the frozen transformed prior. Stress failure cannot be repaired by
    widening the prior after inspecting v4.

## Interpretation

A pass authorizes integration of immutable adaptive-SMC posterior snapshots
into a local particle scenario tree. It does not authorize an LLM efficacy
endpoint by itself.

A failure closes this exact posterior configuration. Diagnose prior support,
likelihood approximation, or rejuvenation using only the banked calibration
artifacts before freezing a successor. Do not compensate by increasing planner
depth or LLM compute.
