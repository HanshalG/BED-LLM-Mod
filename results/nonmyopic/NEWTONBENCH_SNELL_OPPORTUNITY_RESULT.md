# NewtonBench Snell-Law Opportunity Result

## Decision

**Gate failed. No model call is authorized for this construction.**

The audit used the preregistered nine-law prior, 32-action bank, mixed
categorical/Gaussian endpoint, three noise strata, and 9/15-node
Gauss-Hermite calculations. It made zero OpenRouter calls and used no OatML
resources.

## Results

The prior entropy is `2.197225` nats.

| Noise | Greedy root | Depth-2 root | Invalid hypotheses at root | Immediate EIG | Two-step EIG | Entropy after step 1 | Entropy after step 2 | Strict |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| `0.0001` | 0 | 0 | 0 | `2.197225` | `2.197225` | `0` | `0` | No |
| `0.01` | 4 | 4 | 0 | `2.196079` | `2.197225` | `0.001146` | `0` | No |
| `0.1` | 7 | 7 | 2 | `1.528149` | `2.165391` | `0.669075` | `0.031833` | No |

Every noise stratum selected the same root under the greedy and depth-two
objectives, so there is no forced non-myopic tradeoff. At noise `0.1`, the
qualitative invalid branch is active and substantial uncertainty remains after
one experiment, but the same root is still best for both horizons.

All roots are stable across the two quadrature orders. Maximum immediate-EIG
differences are at most `0.003042` nats and maximum depth-two differences are
at most `0.001879` nats, within the frozen `0.005`-nat tolerance.

## Interpretation

The exact invalid/finite outcome channel does create branching, but it does not
create action complementarity. At low noise, a finite angle nearly identifies
the law in one experiment. At noise `0.1`, a second experiment is valuable,
yet the action with the best immediate partition is also the best gateway to
that second experiment.

This separates sequential value from non-myopic value: two observations help,
but planning the first observation beyond greedy EIG does not. The released
unrestricted Snell controls therefore do not support the intended depth claim.
No critical-angle action is inserted after seeing this result, and no support,
noise, threshold, or action-bank repair is attempted.

Together with the sound-speed null, this suggests that NewtonBench is useful
for iterative law discovery but its vanilla-equation tasks do not automatically
form useful non-myopic BED problems. A headline environment needs native
gating, resource constraints, or observations that reveal which specialized
experiment should be run next.

## Reproducibility

- Official source commit:
  `912a4ba5f4356ddd06acc16e44460ca30be4abc2`
- Action-bank SHA-256:
  `4b38f1249b3a4753304eb569c4f43f1e2109b2d796832b64550861d20f97d440`
- Audit JSON SHA-256:
  `a23966dadfa75780bd610ae81eecc220bd7cfa40362500430314e5e25d4103fa`
- Model calls: `0`
