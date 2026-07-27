# DiscoverPhysics Dark-Matter Simulator-Grounded LLM Policy

## Status

Frozen before any response or fresh hidden-map endpoint under this protocol.
This is one nine-call paired policy experiment. It is the only authorized
successor to the passed executable-support V2 serving gate. Failure closes
this exact executable-support/official-likelihood dark-matter route without
prompt repair, response cleanup, partial endpoint, seed swap, threshold
change, reasoning, or model rerun.

## Claim

The LLM generates the open continuous hypothesis supports and
branch-conditioned continuations. The official simulator supplies every
likelihood and trajectory-risk calculation. The primary test is whether
simulator-grounded lookahead over the LLM's own path-dependent supports
selects the oracle-qualified central scout and improves prediction on fresh
hidden maps over myopic, random-root, and fixed-support controls.

The finite hidden-map family, exact likelihoods, branch values, and endpoint
remain invisible to the model.

## Model, Calls, And Budget

- Model: `openai/gpt-5.4`, reasoning disabled, temperature zero.
- One fresh initial executable support; the serving output is discarded.
- Eight isolated branch-conditioned executable-support refreshes.
- Exact requests/HTTP: `9`.
- Projected cost: `$0.14`; hard cap: `$0.25`.
- Strict allowance before responses: `$0.86783155`.
- Last authenticated provider balance: `$33.996547594`.
- Protected reserve: `$25`.
- OatML use: none.

## Shared Roots And Objective Branches

The four roots remain:

- A: northwest target `[-3.182,3.182]`;
- B: center scout `[0,0]`;
- C: southwest target `[-3.182,-3.182]`; and
- D: northeast target `[3.182,3.182]`.

After the fresh eight-map initial support is parsed and compiled, the official
N-body simulator generates each root's active-probe endpoint. Deterministic
weighted two-means creates exactly two objective branches per root.
Representative coordinates, branch probabilities, and exact
Gaussian-likelihood posteriors are computed without an LLM.

Each refresh call receives only its root, representative coordinate,
initial support, exact posterior over that support, and the 25 legal actions.
It returns eight fresh executable weighted hypotheses plus one different
continuation. No self-reported likelihood, EIG, coverage, or readiness score
is requested.

## Selection

Official likelihoods over the initial LLM support determine immediate EIG.
Myopic maximizes it.

For every root and generated-support latent truth, exact Monte Carlo
simulation:

1. samples a root observation;
2. routes it to the nearest objective branch;
3. executes that branch's LLM continuation;
4. updates the branch-refreshed support with the official continuation
   likelihood; and
5. predicts the two official held-out trajectories by posterior mixture.

Lookahead minimizes expected held-out trajectory MSE. Internal seed `24510`
uses 16 root and eight continuation samples per generated truth.

All model outputs, supports, roots, and continuations are checkpointed and
hashed before fresh hidden maps are constructed or scored. There are no model
calls after endpoint access.

## Fresh Hidden Endpoint

Fresh hidden-map seeds `24520,24521,24522,24523` produce 96 maps: 24 maps per
region. Endpoint weights reproduce the region prior
`[.40,.30,.20,.10]`. None of these maps or trajectories was used in the
structural gates or prompts.

Hidden-noise seed `24524` uses eight root and four continuation samples per
map under common random numbers. The endpoint is official posterior-predictive
probe-trajectory MSE on two five-probe experiments through `t=5`.

Policies:

- dynamic lookahead: preregistered center root B if selected internally;
- dynamic myopic: preregistered northeast root D if selected internally;
- seeded random root: fixed root A;
- fixed-support center: root B and the same branch-specific LLM
  continuations as dynamic B, but Bayesian inference remains on the initial
  support instead of regenerated supports.

The fixed-support control isolates the value of branch-conditioned support
regeneration from root and continuation compute.

## Uncertainty And Coverage

Per-map MSEs are retained. A 10,000-sample stratified bootstrap with seed
`24525` resamples within each region and recombines with the frozen region
prior.

Coverage is behavioral rather than textual: for every hidden map, compare
the nearest initial-support held-out trajectory to the nearest
branch-refreshed-support trajectory after the deterministic center outcome.

## Frozen Gates

All mechanics gates are conjunctive:

- exactly nine requests, zero reasoning, cost at most `$0.25`;
- at least 6/8 refreshed supports differ from the initial support;
- at least 3/4 roots have branch-distinct supports;
- center branches choose distinct continuations; and
- every refresh retains at least two regions.

All scientific gates are conjunctive:

- official immediate EIG selects D;
- internal simulator-grounded lookahead selects B;
- B reduces internal generated-world risk by at least `10%` versus D;
- on 96 fresh maps, dynamic B reduces MSE by at least `10%` versus dynamic D;
- the stratified-bootstrap 95% lower bound for per-map `MSE(D)-MSE(B)` is
  positive;
- dynamic B reduces MSE by at least `5%` versus random A;
- dynamic B reduces MSE by at least `5%` versus fixed-support B; and
- nearest refreshed-support trajectory risk improves by at least `5%` over
  nearest initial-support risk.

No directional-only, subgroup-only, symmetry-class, or endpoint-after-failure
pass is allowed. A clean serving run that fails any scientific gate is a
banked null, not authorization for another dark-matter attempt.
