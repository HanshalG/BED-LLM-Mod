# ChemBench M-open Non-Myopic Opportunity Protocol

Date frozen: 2026-08-14 (Europe/London)

## Purpose

Test whether the released ActiveSciBench-Chem simulator contains a strict,
same-utility planning-horizon opportunity before any LLM support generation is
implemented or called. This is a fixed-support oracle yardstick. It cannot by
itself establish an LLM-native result.

The intended descendant follows Murphy's Model Discovery Agent architecture:
an LLM proposes executable rate-law structures after predictive residuals,
numerical evidence weights them, and a non-myopic planner reasons through that
residual-conditioned support transition. The present test asks only whether the
environment is worth that implementation effort.

## Source Binding

Official repository:

```text
https://github.com/scientific-discovery/LLM-AutoSciLab
commit: acf160eb6c96897748dd92b152703b59b74efc05
tree:   e042d418fc30c6c70f6d1c6b0636d43f0c1c0f7a
```

Required source-file SHA-256 values:

```text
autoscilab/oracle/chembench.py
  eba9514d68573b4aa8c6a427606f1d0d7a9c431d8396e69ef4ae0774d2a449de
autoscilab/oracle/compound_domains.py
  0dfd47c1858efacb732f0fbceace3ebd61bb30f864ec777d628fb70f876343f3
autoscilab/oracle/chembench_excluded.py
  defc6c0c5edafe75dffaa366a61298856f413bd465a6e4e3636e9b0c57124003
```

The population is the exact 57-domain active set produced by
`CHEM_DOMAIN_REGISTRY - CHEM_EXCLUDED_DOMAINS`, in registry order. Every domain
must expose `easy`, `medium`, and `hard` parameterizations for `v0`, `v1`, and
`v2`; otherwise the test fails closed.

## Development Disclosure

The protocol was chosen after a development-only screen on `easy/v0`.
Consequently that parameter slice is excluded from validation. With 1% source
noise, four experiments, 1,000 disjoint query points from seed `2026081401`,
and raw mean squared log-rate loss, the seen expected risks were:

| Horizon | Risk | Approx. RMSLE |
| --- | ---: | ---: |
| d1 | 0.02650505 | 0.162804 |
| d2 | 0.01960093 | 0.140003 |
| d3 | 0.01790774 | 0.133820 |

These values selected ChemBench over the explored NeuronBench extension and
must never be presented as prospective evidence.

## Untouched Validation Population

Eight parameter slices are evaluated exactly once:

| Slice | Query seed |
| --- | ---: |
| easy/v1 | 2026081501 |
| easy/v2 | 2026081502 |
| medium/v0 | 2026081503 |
| medium/v1 | 2026081504 |
| medium/v2 | 2026081505 |
| hard/v0 | 2026081506 |
| hard/v1 | 2026081507 |
| hard/v2 | 2026081508 |

This gives 456 paired truth cells (`8 slices x 57 domains`). No validation
response matrix, policy, or loss may be inspected before the implementation,
source verifier, and focused tests are committed.

## Design Space

All assays use the baseline
`(C_A=1, C_I=0, C_B=1, C_P=0, Enz=1, T=310, pH=7)` and change only the named
coordinates. The 18 ordered actions are:

```text
baseline
C_A in {0.02, 0.1, 10, 100}
C_I=50 crossed with C_A in {0.1, 1, 10, 100}
C_B in {0.01, 0.1, 100}
C_P in {10, 20}
T in {278, 368}
pH in {4, 10}
```

An action may be used at most once. Every policy receives a budget of four
experiments.

## Observation Model

For each domain, action, and fixed parameter slice, the source rate function is
multiplied by the source `_secondary_effects(T, pH)` factor. The observation is
then generated with the released multiplicative Gaussian noise level
`sigma_rel=0.01`.

For exact finite-tree evaluation, the continuous observation is reduced to one
of three ordered bins. The two boundaries are the one-third and two-third
quantiles of the 57 prior-predictive action means. Bin probabilities are exact
Gaussian CDF masses under each model. This quantization is part of this audit,
not part of the upstream benchmark and not an assertion that the eventual
method should discard the continuous rate.

The prior is uniform over the 57 fixed point hypotheses. Posterior updates are
ordinary Bayes updates using the three-bin likelihoods.

## Objective And Policies

Each slice gets 1,000 held-out query assays sampled independently from the
official bounds. `C_A`, `C_B`, and `Enz` are log-uniform; all other inputs are
uniform. Query assays are disjoint from the fixed design menu by construction.

For truth `m`, belief `w`, and held-out query set `Q`, terminal loss is

```text
mean_q (sum_j w_j log1p(rate_j(q)) - log1p(rate_m(q)))^2
```

The planning value is its Bayes expectation under the current belief. No
per-query standardization or entropy surrogate is permitted.

Evaluate receding-horizon policies `d1`, `d2`, and `d3`. At every execution
step, each policy solves to its named depth, capped by remaining budget, then
replans after the observation. All policies share the same actions,
likelihoods, priors, terminal utility, four-step execution budget, and tie
breaking by ordered action index.

Report both prior-averaged expected risk and the 456 truth-conditional expected
losses obtained by integrating over each policy's possible observation paths.

## Frozen Gate

All conditions are conjunctive:

1. Source hashes, active-domain count/order, parameter slices, designs, and
   query seeds match this protocol; all values are finite.
2. Aggregate paired mean loss improves by at least 5% for `d2` versus `d1`.
3. Aggregate paired mean loss improves by at least 5% for `d3` versus `d2`.
4. Each successive comparison improves the slice-mean loss on at least six of
   eight untouched slices.
5. `d3` improves over `d1` on every untouched slice.
6. Each successive comparison wins on strictly more than half of the 456
   truth cells after ties within `1e-12` are removed.

Failure closes this exact ChemBench formulation as the primary M-open route;
thresholds, slices, query seeds, and action menus may not be changed after
inspection to rescue it.

A pass authorizes only the next zero-call engineering stage: the typed
executable-law representation, numerical evidence updater, residual trigger,
and cached branch-conditioned proposal interface. It does not authorize a paid
model call or an efficacy claim.

## Required Descendant Controls

Before any policy endpoint, the M-open implementation must freeze and compare:

- dynamic support d1/d2/d3;
- MDA-style myopic residual-triggered expansion;
- call-matched myopic planning over the same cached branch proposals;
- fixed-support d1/d2/d3;
- history-blind residual expansion;
- random valid assays;
- a separately labelled thinking naive baseline.

The primary future endpoint remains held-out predictive loss. Proposal validity,
residual obedience, support novelty, truth coverage, branch-transition fidelity,
planner ranking fidelity, action divergence, calls, tokens, and cost are required
mechanism diagnostics, not replacements for that endpoint.
