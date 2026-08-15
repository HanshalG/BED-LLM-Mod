# ChemBench Policy-Ladder Mechanics V4 Protocol

Date frozen: 2026-08-14 (Europe/London)

This is a prospective successor to the speculative-particle V3 development
screen. It changes the policy construction before any outside-support v4
response is opened.

## Frozen Source and Cohort

- Official source commit/tree: `acf160eb6c96897748dd92b152703b59b74efc05` /
  `e042d418fc30c6c70f6d1c6b0636d43f0c1c0f7a`.
- Nine simple initial candidates use source version `v2`.
- All 48 outside-support candidate/truth structures use untouched source
  version `v4`.
- Slices and proxy-query seeds:
  - `easy/v4`: `2026081701`
  - `medium/v4`: `2026081702`
  - `hard/v4`: `2026081703`
- Four experiments, 18 frozen assays, six assay groups, and 1,000 proxy queries
  per slice remain unchanged.
- All v4 source responses remain sealed until this protocol, implementation,
  tests, producer, and independent replay verifier are committed and pushed.

## Beliefs and Transition

Retain the V3 two-belief state:

- inference belief: live/reserve executable support plus the scalar M-open
  alarm;
- speculative belief: uniform oracle prior over all 48 eligible outside worlds.

The selected truth ID is absent from policy state. A branch updates speculative
weights by the frozen categorical likelihood and expands inference support with
the cached registry-oracle proposal. The oracle proposes at most four missing
structures by complete-history likelihood. This is a non-deployable mechanics
ceiling.

## Policy Levels

`d1` is the one-step dynamic-support policy. For `k > 1`, `d_k` is one exact
finite-budget policy-improvement step over `d_(k-1)`:

```text
d_k(s,t) = argmin_a E[J_(d_(k-1))(T(s,a,Y),t-1)]
```

`J_(d_k)` always evaluates execution of the same `d_k` policy through the full
remaining experiment budget. It is not a truncated leaf score and not a
receding fixed-horizon plan. Each state considers one highest-speculative-
variance assay per group, at most six actions, for every policy level. Therefore
the predecessor action remains available to its successor.

The submitted forecast remains the inference-belief Bayes forecast. Terminal
risk is speculative expected squared log-rate error on the public proxy set.

## Controls

- old speculative receding d1/d2/d3, descriptive only;
- policy-ladder d1/d2/d3, primary;
- call-matched d1 replay after d3 has populated the complete immutable proposal
  bank;
- scalar-unknown V2, fixed-support, scripted residual, and history-blind
  diagnostics from prior protocols.

## Conjunctive Gate

1. Source/version/privacy, finite/normalization, zero-call, and zero-cost checks
   pass.
2. For every slice and policy level, planned terminal risk equals uniform
   truth-conditional replay within `1e-10`.
3. Model-based risk is non-increasing d1 to d2 to d3 on every slice within
   `1e-12`.
4. Aggregate d2 risk is at least 5% below d1.
5. Aggregate d3 risk is at least 5% below d2.
6. Each successive comparison has more paired truth-cell wins than losses at
   practical tolerance `1e-6`.
7. d2 differs from d1 at the root on at least one slice.
8. d3 differs from d2 at the root on at least one slice.
9. Call-matched d1 replay is exact and adds zero producer misses.
10. A banked-proposer replay independently reproduces every policy root,
    planned value, truth loss, comparison, and gate.

Failure closes this exact categorical/fixed-parameter policy-ladder mechanics.
It does not authorize threshold changes, v5 responses, or a paid LLM gate.

## Development Disclosure

The already-open v3 prototype produced aggregate d1/d2/d3 terminal MSE
`.03652561/.03256128/.02461982`, reductions of `10.85%` and `24.39%`.
These values motivated this policy correction but do not count toward the v4
gate.
