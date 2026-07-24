# ClinDiag Joint Patient-Response Function Gate Result

Date: 2026-07-24

Status: **qualification failed; no likelihood-fidelity stage, structural gate,
planner, or holdout followed.**

## Frozen Qualification Result

| Criterion | Required | Observed | Pass |
|---|---:|---:|---:|
| Physical requests | 10 | 10 | yes |
| Reasoning tokens | 0 | 0 | yes |
| Parsed supports | 2 of size 12 | 2 of size 12 | yes |
| Safe six-query sets | 2/2 | 2/2 | yes |
| Literal target leaks | 0 | 0 | yes |
| Duplicate yes/no labels | 12/12 | 12/12 | yes |
| Fully valid audited query pairs | 12/12 | 11/12 | **no** |

Joint response generation fixed the previous missingness and identity problems:

- no answer exposed absent chart data;
- all 12 original and duplicate yes/no labels matched;
- findings and provenance were stable under the full-function duplicate.

One load-bearing factual error remained. For the `rare214` T-B+ severe combined
immunodeficiency case, both jointly generated functions answered:

> Eczema is absent on skin examination.

and labeled the fact `recorded`. The independent hidden-case audit rejected both the
original and duplicate as case-inconsistent.

## Interpretation

The error is not serving noise: exact duplicate generation reproduced it. Joint
generation made the environment stable but stably wrong. Because actual deployed
observations define the trajectory, a single fabricated patient fact can redirect
belief regeneration and any non-myopic policy. The frozen all-rows criterion is
therefore appropriate and is not relaxed to 11/12.

This is the third ClinDiag gatekeeper construction to fail environment validity:

1. four-way GPT-5.4 Mini responses were often inconsistent;
2. independent GPT-5.4 binary calls exposed chart missingness;
3. joint GPT-5.4 binary functions removed instability but retained factual
   hallucination.

## Cost

- 10 requests;
- 13,455 prompt tokens and 2,539 completion tokens;
- zero reasoning tokens or retries;
- `$0.06769050`;
- conservative remaining project balance: `$23.82953322`.

## Consequence

The exact LLM-generated patient-environment line stops. The next route should keep
ClinDiag's observed environment deterministic and use the LLM only where it is
scientifically load-bearing: open-world hypothesis generation, history-conditioned
filtering, and semantic likelihood prediction.

A valid distinct construction is a fixed-slot evidence environment. Every case exposes
the same generic action labels (for example history slot 1, examination slot 1,
laboratory slot 1), and selecting an action reveals a stored text chunk or a fixed
empty result. This avoids retrospective procedure-menu leakage and patient-simulator
hallucination while preserving semantic, path-dependent LLM beliefs.
