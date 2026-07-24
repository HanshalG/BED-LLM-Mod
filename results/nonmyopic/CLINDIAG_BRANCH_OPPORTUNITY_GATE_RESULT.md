# ClinDiag Branch-Opportunity Gate Result

Date: 2026-07-24

Status: **opportunity gate failed; no target-blind planner or holdout is authorized.**

## Protocol

Twelve fresh ClinDiag cases, balanced six challenging and six rare, each exposed five
nonempty native evidence actions: medical history, physical examination, laboratory
tests, imaging, and other tests. From an initial 12-diagnosis support, non-reasoning
GPT-5.4 regenerated supports after all five one-step actions and all 20 ordered pairs.

One exact second-step prompt was repeated per case as a serving-noise control. The true
diagnosis entered only post-generation GPT-5.4 Mini semantic measurement. All 336
expected physical requests completed with zero retries, reasoning tokens, forced exits,
or runtime failures.

## Frozen Gate

| Criterion | Required | Observed | Pass |
|---|---:|---:|---:|
| Complete cases | 12 | 12 | yes |
| Initial coverage | at most 5 | 2 | yes |
| One-step spread at least 0.20 | at least 8 | 9 | yes |
| Reverse-order gap at least 0.15 | at least 5 | 7 | yes |
| Non-myopic gap at least 0.10 | at least 4 | 1 | **no** |
| Mean non-myopic gap | at least +0.05 | +0.025 | **no** |
| Mean oracle two-step gain | at least +0.10 | +0.025 | **no** |
| Mean identity score gap | at most 0.05 | 0.0917 | **no** |
| Maximum identity score gap | at most 0.15 | 1.00 | **no** |

The pre-formal amendment correctly retained exact-name Jaccard as descriptive only;
its observed mean was `0.4559`.

## Diagnosis

The gate separates action dependence from useful non-myopia:

- nine cases had meaningful one-step score spread;
- seven had a meaningful reverse-order score difference;
- but the best one-step action already attained the best two-step score in 11/12
  cases.

Only fetal alcohol syndrome had a genuine oracle non-myopic gap:

- best one-step score: `0.60` after history;
- best continuation after that greedy action: `0.60`;
- best ordered pair: laboratory tests then history, score `0.90`;
- non-myopic gap: `+0.30`.

Every other case had zero gap, even when the first action of an oracle-maximizing
sequence differed from the greedy action. The coarse native evidence blocks usually
contained enough signal for one-step support recovery, leaving no load-bearing role for
horizon.

The identity control also caught a severe serving-noise event for delayed-onset
heparin-induced thrombocytopenia. Two identical laboratory-then-examination prompts
scored `1.0` and `0.0`; one support contained heparin-induced thrombocytopenia and the
other omitted it. This demonstrates that some apparent reverse-order differences from
single-sample regeneration are not attributable to path order.

## Cost And Integrity

- Serving smoke: 10 requests, zero reasoning, `$0.02974350`.
- Formal opportunity gate: 336 requests, 260,306 prompt tokens, 58,119 completion
  tokens, zero reasoning, `$1.30918475`.
- Complete branch-opportunity line: `$1.33892825`.
- Complete ClinDiag line through this gate: `$1.54945375`.
- Project-ledger spend: `$46.36670922`, leaving `$24.01809347` conservatively.
- The live OpenRouter endpoint reported `$24.16188572` remaining immediately after
  the run; the project ledger remains the stricter limit.
- The 60-case generator holdout remained untouched.

## Consequence

The external ClinDiag result now establishes reliable full-workup support recovery but
not a native-block non-myopic opportunity. The exact five-action route stops here:
there will be no threshold repair, replacement cases, target-blind ranker, policy, or
holdout run.

A materially distinct future route would need both:

1. finer evidence actions so no single action usually saturates support value;
2. a replicated or aggregated support estimator whose semantic truth score is stable
   under identical prompts.

Those changes require fresh cases and a new preregistration; they cannot be presented
as a repair of this null.
