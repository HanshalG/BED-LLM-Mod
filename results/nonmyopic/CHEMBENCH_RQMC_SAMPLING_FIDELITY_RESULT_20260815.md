# ChemBench RQMC Sampling Fidelity Result

Date: 2026-08-15 (Europe/London)

## Binding

- Result: `results/nonmyopic/chembench_rqmc_sampling_fidelity/result.json`
- Result SHA-256:
  `a966d5cf4984c9907649a0dae5d6bb8a19982f942c83f463f7ec61e4e2d439f2`
- Protocol SHA-256:
  `312cbc70d4f34df777e0cc5f35afc4c7779eb9073341b187ab4c8c6f8d939791`
- Implementation SHA-256:
  `8261f6e9e212665fb3e905e45c700f6fdbd70008bff7061b332b50312e70e15a`
- IID predecessor SHA-256:
  `d71964ff247bc80408b0b5c78c4b168c5117ece372446990c4a5dfd17b42c0f6`

The complete 36-case result finished in 26.18 seconds with no LLM, API,
network call, endpoint, or paid resource.

## Frozen Decision

The gate **failed**. Full-panel RQMC does not open depth.

| 256 scramble | Median rho | Fraction rho >= .8 | Regret <= 3% | Mean regret |
|---:|---:|---:|---:|---:|
| 1 | 0.7327 | 36.11% | 100% | 0.159% |
| 2 | 0.7124 | 33.33% | 100% | 0.153% |
| 3 | 0.7466 | 38.89% | 100% | 0.152% |
| 4 | 0.7466 | 36.11% | 100% | 0.111% |
| Required, each | >= 0.90 | >= 90% | >= 90% | <= 1% |

All pooled top-action regret conditions passed. Component-bank regret passed
for most but not every smaller-prefix replicate. The 256 selected-action
agreement was high at 93.52%, but the four-shift ensemble still had median rho
0.7359 and only 38.89% rank coverage.

## Diagnosis

This RQMC construction is inappropriate for the raw SMC particle array.
Selecting an equal-weight particle through a Sobol CDF treats particle index as
a smooth coordinate. SMC particle ordering is arbitrary and includes
resampling structure, so regular CDF strata alias unrelated parameter states
with the second-coordinate Gaussian noise. Independent digital shifts often
produce the same or similarly biased action ordering, explaining high
replicate agreement without reference fidelity.

Do not repair this failed full-panel gate by permuting particles, adding
scrambles, or increasing samples. The frozen successor decision is a
prospective numerical action shortlist. Compute a cheap moment-matched expected
task-variance-reduction proxy from the root posterior, select a fixed-size top
set without reference outcomes, then audit both shortlist oracle regret and
shared-IID-CRN ranking on that set.

No depth or LLM call is authorized.
