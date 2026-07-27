# NewtonBench Remaining-Domains Opportunity Sweep

Date: 2026-07-28

## Status

Frozen before computing any new prediction matrix, EIG, posterior entropy, root
selection, or depth-two value. This is a zero-call structural screen over the ten
NewtonBench domains not previously audited. It is not an LLM result.

## Source And Scope

- official repository: `HKUST-KnowComp/NewtonBench`;
- pinned commit: `912a4ba5f4356ddd06acc16e44460ca30be4abc2`;
- system: each module's `vanilla_equation`;
- prior: uniform over all nine released `easy/medium/hard x v0/v1/v2` laws;
- excluded because they were already audited: `m4_snell_law` and
  `m8_sound_speed`;
- included exactly:
  - `m0_gravity`;
  - `m1_coulomb_force`;
  - `m2_magnetic_force`;
  - `m3_fourier_law`;
  - `m5_radioactive_decay`;
  - `m6_underdamped_harmonic`;
  - `m7_malus_law`;
  - `m9_hooke_law`;
  - `m10_be_distribution`;
  - `m11_heat_transfer`.

The policy-facing benchmark prompt does not disclose difficulty or law version. The
released law registry is used only to establish whether a structural opportunity
exists. A later LLM policy may not receive or use that registry.

## Frozen Action Banks

Each domain receives exactly `32` Latin-hypercube actions. Dimension permutations and
within-bin jitter use an independent `random.Random(24420 + module_index)`. Parameters
use the ranges sampled by the pinned module's own evaluation code:

| Module | Parameters and ranges |
|---|---|
| `m0` | `mass1,mass2: log[1,1000]`; `distance: log[1,10]` |
| `m1` | `q1,q2,distance: log[0.1,10]` |
| `m2` | `current1,current2,distance: log[0.001,0.1]` |
| `m3` | `k: log[0.1,10]`; `A: log[1e-4,1e-2]`; `delta_T: log[10,1000]`; `d: log[0.01,1]` |
| `m5` | `N0: log[1,100]`; `lambda_constant: log[0.001,0.1]`; `t: log[0.01,10]` |
| `m6` | `k: log[100,10000]`; `m: log[0.1,10]`; `b: log[0.01,1]` |
| `m7` | `I_0: log[100,2000]`; `theta: linear[1e-6,pi/2]` |
| `m9` | `x: log[0.001,1]` |
| `m10` | `omega: log[1e8,1e10]`; `T: log[10,1000]` |
| `m11` | `m: log[0.001,1000]`; `c: log[100,10000]`; `delta_T: log[10,1000]` |

Action order is the generated row order. Every bank SHA-256 is recorded before its
prediction matrix is evaluated. No action may be added, removed, clipped, or locally
perturbed after outcomes.

## Observation And Planning Model

For every finite law output `mu`, the observation distribution is the benchmark's
relative Gaussian model:

```text
Normal(mu, max(abs(mu * noise_level), 1e-9))
```

A non-finite law output is an exact categorical `invalid` observation. Finite and
invalid outcomes are handled jointly; invalid outputs are not dropped. The declared
noise strata are exactly `.0001`, `.01`, and `.1`.

The prior is uniform and entropy is measured in nats. Expected entropies use
deterministic Gauss-Hermite quadrature with `15` nodes for the primary result and `9`
nodes for convergence. The second action is selected adaptively from the same
domain-specific bank after every first observation. Action reuse is allowed.

Greedy maximizes immediate EIG, then depth-two EIG, then bank order. Non-myopic
maximizes depth-two EIG, then immediate EIG, then bank order. Numerical ties use
`1e-12`.

## Development Candidate Gate

A domain/noise cell is a development candidate only if all hold:

1. greedy and non-myopic roots differ;
2. non-myopic immediate EIG is at least `.01` nats lower;
3. non-myopic depth-two EIG is at least `.01` nats higher than the greedy root with
   its own optimal observation-conditioned continuation;
4. both margins retain their signs with 9-node quadrature;
5. both quadrature orders select the same roots; and
6. every immediate and depth-two value differs by at most `.005` nats between
   quadrature orders.

All thirty cells are reported. If none passes, the sweep closes before model use.

## Independent Action-Bank Confirmation

If multiple development cells pass, select exactly one by:

1. largest 15-node depth-two gain;
2. then largest immediate sacrifice;
3. then lowest module index;
4. then noise order `.0001`, `.01`, `.1`.

The selected module/noise pair is recomputed on an independent `32`-action
Latin-hypercube bank using seed `24520 + module_index` and the same ranges. Confirmation
requires the same six conditions above. Root row IDs need not match across independent
banks, but both sacrifice and terminal gain must again be at least `.01` nats.

Failure closes the selected domain and the full sweep. No runner may substitute a
second-ranked development domain after confirmation failure.

## Conditional LLM-Native Route

A confirmed structural pass authorizes only a separate preregistered serving smoke in
which the LLM:

- generates an open semantic law support without registry access;
- proposes executable experiments within the unchanged native ranges;
- predicts or regenerates path-dependent beliefs from observations; and
- chooses between matched myopic and depth-two objectives on the same generated
  support/action bank.

The official simulator supplies observations and terminal law prediction error. A
fixed-registry planner remains an oracle yardstick, never the deployed policy.
Reasoning is reserved for a naive-thinking baseline.

There is no protected OpenRouter reserve. The structural sweep itself makes zero model
calls. OpenRouter is the only permitted paid backend; OatML, Slurm, and SSH are
prohibited.

