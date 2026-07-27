# DiscoverPhysics Extra-Dimensions Structural Confirmation

## Status

Frozen before running the official-simulator confirmation. This is a zero-call
mechanics gate, not an LLM-policy result. Failure closes this candidate before
any OpenRouter request.

## Development Boundary

A seed-`24670` search evaluated `50` candidate priors with deterministic
Gauss-Hermite quadrature. The search used three official pairwise kernel
families and selected the highest total-EIG-margin candidate that passed
fixed `.03`-nat immediate-sacrifice and total-gain screens.

Development artifact SHA-256:
`39ea57dc29e55e21ca018988e3e6959435fc888516a952cbcdcda3102006e41f`.
Search script SHA-256:
`32591a1f17a3fb57c31042c850c88b86237a08511eceb2648870cee7755ffbe3`.

The physical trajectory statistic and its scale were also chosen during
development. No result from the official `NBodySampler` confirmation has been
observed.

## Frozen Prior

Official DiscoverPhysics commit:
`33b7fa9df96de9c35744efd181ca7e5a8dd60ad5`.

The prior contains `18` executable radial force laws:

- six Kaluza-Klein laws with compact radii
  `.18,.25,.35,.50,.72,1.05`, using `60` images in each direction and
  total prior mass `.40`;
- six 2D Yukawa laws with screening lengths
  `.55,.80,1.15,1.70,2.70,4.50`, total prior mass `.35`;
- six 2D Riesz laws with fractional exponents
  `.30,.42,.54,.66,.78,.90`, total prior mass `.25`.

Members are uniform within family. Each member's strength is calibrated to a
Poisson `1/(2 pi r)` reference:

- Kaluza-Klein at radius `.75`, family offset
  `.7177711921002633`;
- Yukawa at radius `.75`, family offset
  `.913222185103666`;
- Riesz at radius `8.0`, family offset
  `.9803504883305367`.

The policy is not shown this enumeration in any later LLM gate. Here it is
only the oracle instrument used to prove that an adaptive opportunity exists.

## Frozen Experiments

The action bank consists only of initial probe radii
`[2.4,3.6,5.5,8.0]`. Every experiment uses:

- one fixed central source and four noninteracting probes;
- source strength `p1=1`, probe inertia `p2=1`;
- initial position `[radius,0]`, initial velocity `[0,0]`;
- official `NBodySampler`, `yoshida4`, `dt=.005`, softening `.05`;
- duration `1.0`.

For hypothesis `h` and action radius `r`, the scalar observation mean is

`0.4 * log(max(r - x_h(1.0), 1e-12))`.

Observed values have independent Gaussian noise with standard deviation
`.075`. This is a multiplicative-precision displacement sensor. Exact
24-point Gauss-Hermite quadrature computes immediate and adaptive depth-two
EIG.

The external prediction feature is `0.4 * log(force_h(r))` at `96`
log-spaced radii from `.3` through `9.0`. Posterior-mean squared error is
averaged over the prior, root observations, and adaptively selected
continuations.

## Frozen Gates

All conditions must pass:

1. Every official trajectory and transformed mean is finite and every radial
   displacement is positive.
2. Official myopic root is radius `5.5`.
3. Official adaptive depth-two root is radius `2.4`.
4. The depth-two root gives up at least `.10` immediate nats.
5. The depth-two root gains at least `.05` total EIG nats.
6. Its held-out posterior-predictive MSE is at least `20%` below the myopic
   root.
7. Re-evaluation at 16 and 32 quadrature points returns the same two root IDs,
   at least `.04` total-EIG gain, and at least `15%` risk reduction.
8. A separate NumPy Yoshida implementation agrees with every official
   displacement mean to maximum absolute error at most `1e-7`.

There is no candidate substitution, action change, support change, scale
change, threshold relaxation, or partial pass after the official result.

## Authorization

A full pass authorizes only a separately frozen, at-most-`$0.75` OpenRouter
mechanics smoke in which a non-reasoning LLM generates executable initial and
branch-conditioned force-law supports. Fixed numerical actions and exact
simulator likelihoods remain machine-owned. Reasoning is reserved for the
naive baseline.

OpenRouter cost for this gate: `$0`. OatML/cluster use: none.
