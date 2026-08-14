# ChemBench M-open Mechanics V1 Protocol

Date frozen: 2026-08-14 (Europe/London)

This protocol instantiates the zero-call gate in
`CHEMBENCH_NONMYOPIC_MOPEN_ARCHITECTURE_PROTOCOL_20260814.md`. It may use the
pinned source registry as an oracle ceiling, but it makes no model or network
calls and opens no `v4` or `v5` state.

## Immutable Source and Cohort

- Source commit: `acf160eb6c96897748dd92b152703b59b74efc05`.
- Source tree: `e042d418fc30c6c70f6d1c6b0636d43f0c1c0f7a`.
- All 57 active source worlds are paired truth cells.
- Untouched mechanics slices and proxy-query seeds:
  - `easy/v3`: `2026081601`
  - `medium/v3`: `2026081602`
  - `hard/v3`: `2026081603`
- Each proxy endpoint contains 1,000 assays sampled from the official bounds.
- No `v3` response matrix may be constructed before this protocol and the
  implementation with synthetic/adversarial tests are committed and pushed.

## Approximation Boundary

This is a structural mechanics test, not efficacy. It uses each source world's
released fixed `v3` parameter point as the fitted representative of that
structure. It uses the same three-bin likelihood convention and 18 public
assays as the passed opportunity audit. Future serving and efficacy must restore
raw rates and fit parameters from observed data.

## Initial Support

Start from the nine active simple mechanisms:

```text
c0_michaelis_menten
c1_competitive_inhibition
c2_product_inhibition
c3_arrhenius_temperature
c5_pingpong_bisubstrate
c6_uncompetitive_inhibition
c7_substrate_inhibition
c8_hill_cooperativity
c9_noncompetitive_inhibition
```

The other 48 active structures are outside the initial support. Live support is
capped at 12 and reserve support at 12. Model weights are recomputed from the
entire branch history whenever support changes.

Use an initial unknown-support score of `0.35` with a uniform three-bin
predictive. Its posterior score is updated prequentially and reported
separately. Endpoint forecasts renormalize represented-model weights; a fixed
unknown-mass penalty equal to initial represented-support Bayes risk is included
in planning leaves.

## Assay Shortlist

Partition the 17 non-baseline public assays into six groups by the principal
changed input: `C_A`, `C_I`, `C_B`, `C_P`, `T`, and `pH`. At each belief state,
select the available assay with maximum represented posterior predictive
variance from each group. This yields at most six root actions without using a
truth identifier or endpoint labels.

Progressive widths are six actions at the root, three at child nodes, and two
at grandchild nodes. Within a narrower node, retain the actions with greatest
represented predictive variance. Ties use public assay order.

## Proposal Kernels

Every kernel is a deterministic pure function of branch history and seed and is
wrapped in an immutable cache.

### Registry Oracle

The oracle may inspect all source structures but never the selected truth
identifier. After appending a hypothetical observation, rank missing structures
by complete-history categorical log likelihood and propose the top four. Ties
use active-domain order. This is a non-deployable ceiling.

### Scripted Residual Kernel

The scripted kernel sees only the latest assay group, categorical residual
direction, tried structures, and public history. It proposes up to four entries
from a frozen group/outcome dictionary containing simple compositions and
active novel exemplars. It cannot enumerate or rank the full source registry.
Numerical code subsequently scores accepted candidates on the entire history.

### Ablations

- Fixed support returns no proposal.
- History-blind refresh returns the same number of candidates from a seeded
  static ordering independent of action and observation.
- An invalid kernel used in tests returns out-of-range and duplicate IDs; all
  invalid entries must be rejected without mutating the parent state.

The mechanics always requests a proposal after each simulated observation. This
is the expansion stress test explicitly permitted by the parent protocol; the
future LLM trigger remains separately gated.

## Update and Objective

Known-model scores use a uniform structure prior over discovered models times
the complete-history likelihood. Unknown support uses its `0.35` prior and a
uniform likelihood. Retain the 12 highest-scoring represented models as live;
retain the next 12 as reserve. All probability vectors must be finite and
normalized.

The leaf score is represented posterior Bayes risk over the 1,000 proxy
log-rate targets plus the fixed unknown-mass penalty. Planning exactly sums the
three categorical branches. Policies execute four assays with receding horizon
one, two, or three. Realized truth replay draws from each truth world's exact
categorical likelihood and scores its proxy log-rate MSE.

Run d3 first to populate the proposal cache. The call-matched myopic replay then
uses the same cache and depth-one leaf objective. Cache misses during its root
tree are forbidden. Dynamic d1 and the call-matched myopic replay must select
the same actions; this control validates call matching rather than creating a
different scientific arm in zero-call mechanics.

## Conjunctive Gate

The oracle result passes only if:

1. source bindings, dimensions, finite values, and normalization checks pass;
2. model/API calls and cost are exactly zero;
3. no policy state, cache key, or public replay record contains a truth ID;
4. call-matched myopic has zero cache misses and exactly matches dynamic d1;
5. d2 aggregate terminal MSE is at least 5% below d1;
6. d3 aggregate terminal MSE is at least 5% below d2;
7. each successive comparison wins more paired truth cells than it loses;
8. d2 and d1 root actions differ on at least one of three slices;
9. fixed and history-blind kernels never receive oracle candidates;
10. invalid proposals leave the parent state unchanged;
11. a producer-independent replay of the result and transition bank reproduces
    root choices, per-truth losses, aggregate comparisons, and cache counts.

The scripted kernel is descriptive in V1. Its support coverage, monotonicity,
and gap to the oracle determine whether to improve the dictionary or proceed
directly to a prospectively frozen LLM semantic gate, but cannot rescue a failed
oracle gate.

V1 is terminal after its one exact command. Failure authorizes a prospective V2
mechanics change on untouched `v4` only; it does not authorize inspecting or
reusing partial `v3` results.
