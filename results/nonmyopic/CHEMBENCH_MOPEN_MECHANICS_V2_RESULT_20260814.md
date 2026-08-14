# ChemBench M-open Mechanics V2 Result

Date: 2026-08-14 (Europe/London)

## Decision

V2 is a verified `failed_closed` scientific null. The dynamic registry-oracle
support transition substantially outperformed fixed, history-blind, and
scripted support, but its depth effect did not generalize from opened `v2`
development states to the frozen outside-support `v3` cohort.

No LLM semantic gate or paid call is authorized by this result.

## Immutable Artifacts

- Implementation commit:
  `d01d3c42f8ae44e847290e7569035efb698f5524`
- Result SHA256:
  `e4a9527a3016953ed88a2cf65f8727a9633f59b734e2fb17de6aa23fc2ab6948`
- Transition-bank SHA256:
  `b041b719734cbf481c2b1d77191dfc6db8d54fbd8af2f081f717b79dc840d754`
- Independent verification SHA256:
  `f5c2d45ecbf977ea7236e4e1f56dd016cfcdb3464a090b193aab4f8fe0cc1757`
- Source commit:
  `acf160eb6c96897748dd92b152703b59b74efc05`
- Truth cohort: 48 structures outside the nine-model initial support, paired
  across `easy/v3`, `medium/v3`, and `hard/v3`.
- Model/API calls: `0`.
- Cost: `$0`.

## Frozen Gate Result

| Horizon | Aggregate terminal log-rate MSE |
| --- | ---: |
| d1 | 0.04408892 |
| d2 | 0.04340479 |
| d3 | 0.06044743 |

| Comparison | Relative reduction | Practical wins / ties / losses |
| --- | ---: | ---: |
| d2 vs d1 | 1.55% | 35 / 81 / 28 |
| d3 vs d2 | -39.26% | 39 / 74 / 31 |
| d3 vs d1 | -37.10% | 46 / 59 / 39 |

Both paired-majority conditions pass, but both required 5% aggregate reductions
fail. d1 and d2 choose the same root on all three slices, so the root-change
condition also fails. Exact producer-independent replay reproduces every root,
per-truth loss, aggregate, gate, and cache count.

## Slice Results

| Slice | d1 root / MSE | d2 root / MSE | d3 root / MSE |
| --- | --- | --- | --- |
| easy/v3 | `C_I=50,C_A=10` / 0.03038 | same / 0.03215 | `C_B=0.01` / 0.03160 |
| medium/v3 | `C_P=20` / 0.05815 | same / 0.05266 | same / 0.05324 |
| hard/v3 | `T=278` / 0.04373 | same / 0.04540 | `C_A=0.02` / 0.09650 |

The hard d3 root causes most of the aggregate reversal.

## First-Link Diagnostic

After opening the terminal result, every one of the six frozen root candidates
was forced and evaluated with the same continuation policy. Spearman
correlations compare the planner's predicted root value with realized terminal
risk; lower is better for both.

| Slice | d1 rho | d2 rho | d3 rho | d3 top-1 regret |
| --- | ---: | ---: | ---: | ---: |
| easy/v3 | -0.429 | -0.486 | -0.257 | 0.01164 |
| medium/v3 | 0.257 | 0.029 | 0.429 | 0.01736 |
| hard/v3 | 0.886 | 0.029 | -0.314 | 0.06031 |

For hard d3, the planner predicts `C_A=0.02` as best, but its realized risk is
0.09650. `C_I=50,C_A=1` realizes 0.03619 and `T=278` realizes 0.03740.

## Localization

The failure is not inability to recover the source structure. Under every
horizon and slice, the registry oracle places the true structure in discovered,
represented, and live support by the four-experiment endpoint with expected
coverage effectively 1.0. Mean terminal represented truth probability is:

| Slice | d1 | d2 | d3 |
| --- | ---: | ---: | ---: |
| easy/v3 | 0.780 | 0.755 | 0.816 |
| medium/v3 | 0.751 | 0.756 | 0.781 |
| hard/v3 | 0.670 | 0.702 | 0.696 |

The failure occurs earlier, in action valuation. Missing support is represented
by a scalar uniform three-bin `unknown` component. It can carry alarm mass, but
it has no model-specific assay predictions or proxy-endpoint semantics.
Consequently the planner's represented-support Bayes risk is not the expected
truth loss under its own support-generation process. Deeper search optimizes a
miscalibrated surrogate more strongly, reproducing the central failure mode
seen in earlier StrategyEIG work.

## Architecture Consequence

The successor must replace scalar unknown support during planning with a
weighted speculative executable-model particle set. Each speculative particle
provides assay likelihoods and endpoint predictions even before promotion to
live inference support. A branch observation updates both:

1. the planning posterior over speculative world particles; and
2. the live/reserve inference support through the cached proposal transition.

Leaf value must average terminal live-belief forecast error against the
speculative posterior particles. The selected truth identifier remains absent;
the oracle mechanics uses the complete 48-world prior only as a non-deployable
ceiling. The future LLM route must generate and calibrate this speculative bank,
which makes proposal quality the irreducible first-link gate.

Use untouched outside-support `v4` for this successor mechanics. Reserve `v5`
by a prospectively frozen domain split for semantic development and efficacy
confirmation. V2 and its exact `v3` interface remain closed.
