# NewtonBench Sound-Speed Opportunity Result

## Decision

**Gate failed. No model call is authorized for this construction.**

The audit used the exact preregistered nine-law prior, 32-action bank, three
noise strata, and 9/15-node Gauss-Hermite calculations. It made zero
OpenRouter calls and used no OatML resources.

## Results

The prior entropy is `2.197225` nats.

| Noise | Greedy root | Depth-2 root | Greedy immediate | Depth-2 immediate | Greedy two-step | Best two-step | Immediate sacrifice | Two-step gain | Strict |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| `0.0001` | 0 | 0 | `2.197225` | `2.197225` | `2.197225` | `2.197225` | `0` | `0` | No |
| `0.01` | 0 | 0 | `2.197225` | `2.197225` | `2.197225` | `2.197225` | `0` | `0` | No |
| `0.1` | 31 | 18 | `2.176154` | `2.095929` | `2.197182` | `2.197204` | `0.080225` | `0.000022` | No |

At noise `0.1`, the roots are stable across both quadrature orders and all
maximum value differences are below the preregistered `0.005`-nat numerical
tolerance. The failure is scientific rather than numerical: the non-myopic
root's terminal advantage is only `2.224e-5` nats, far below the required
`0.01` nats.

The selected `0.1` actions were:

- greedy root 31: `gamma=1.355864`, `T=468.603246`,
  `M=0.005571604`;
- depth-two root 18: `gamma=1.699159`, `T=36.678402`,
  `M=0.003008475`.

## Interpretation

At noise `0.0001` and `0.01`, the best single experiment recovers the complete
nine-law prior entropy to the reported precision. At noise `0.1`, both
two-step policies recover more than `99.998%` of prior entropy. The first
action can change when terminal value is considered, but that change has
negligible utility because the greedy root plus its own adaptive continuation
already saturates the task.

This module is therefore unsuitable for the intended headline claim. Its
continuous controls and widely separated released transformations make law
identification too easy for one unrestricted experiment. Adding artificial
query constraints, restricting the known support after seeing this result, or
interpolating a favorable noise level would change the task post hoc and is
not pursued.

NewtonBench may still contain a useful domain whose action effects specialize
across hypotheses. Any such domain requires a fresh zero-call preregistration;
this sound-speed interface receives no prompt repair or paid retry.

## Reproducibility

- Official source commit:
  `912a4ba5f4356ddd06acc16e44460ca30be4abc2`
- Action-bank SHA-256:
  `edc95b9a080f4ab13d2dba0b6cb857e77b837b044b5940ab2b830d338d3b3c9a`
- Audit JSON SHA-256:
  `7752d3df543e0df810d3556adecec1627fe88a7dab6d677da298a37b1e10d4a3`
- Model calls: `0`
