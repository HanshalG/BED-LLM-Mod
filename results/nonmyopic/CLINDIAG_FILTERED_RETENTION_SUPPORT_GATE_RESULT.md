# ClinDiag BED-LLM Filtered-Retention Gate Result

Date: 2026-07-24

Status: **failed; the exact zero-retry interface is not authorized for scaling.**

## Frozen Gate

The preregistered seed-`24299` gate used fresh cases `11388546` and `rare203`,
one stored `lab_1` update, threshold `0.20`, 12-diagnosis supports, and no structured
retries. Old diagnoses were checked only against the latest observation. New
diagnoses were generated without the old support and checked against initial
presentation plus `lab_1`.

The run completed exactly 14 requests with zero reasoning, retries, parser failures,
or source-target leaks. Cost was `$0.04314350`.

## Result

| Case | Old retained | Valid new | Final size | Replay valid/final | Truth score initial -> final/replay |
|---|---:|---:|---:|---:|---:|
| `11388546` | 12/12 | 7 | 12 | 5/12 | .92 -> .92/.92 |
| `rare203` | 0/12 | 8 | 8 | 6/6 | 0 -> 1/1 |

The frozen pass rule failed:

- `11388546` pruned no old hypotheses and therefore admitted no replacements;
- `rare203` pruned every old hypothesis but the zero-retry candidate pass supplied
  only 8 valid diagnoses originally and 6 on replay, below the required 12.

This split is clinically coherent. The bronchospasm case's blood-pressure observation
was weakly discriminative: all old likelihoods were `.28` to `.72`. The adrenal case's
laboratory panel contradicted every old puberty/neuromuscular diagnosis: all old
likelihoods were `.01` to `.18`. Its newly generated support recovered the exact
CYP17A1/17-alpha-hydroxylase-deficiency truth in both paths.

## Audit Defect

The semantic judge reported directional set overlap `1.0` for the adrenal case. That
is false. The final supports contained 8 and 6 diagnoses and had only one exact string
in common. Manual semantic comparison finds several related steroidogenesis diagnoses,
but not all items have same-disease or standard-synonym matches in both directions.
The judge explanation referred only to the shared true diagnosis and appears to have
answered truth coverage instead of set overlap.

Therefore the nominal overlap gate is not evidence of support stability. The raw
responses are retained for audit.

## Interpretation And Decision

Filtered retention avoids both earlier extremes on the informative case: it removes
the stale support and recovers an omitted truth from deterministic stored evidence.
That is the first positive path-dependent support-transition signal in this ClinDiag
line. It is not yet a usable planner interface:

1. an uninformative observation can correctly cause no support movement;
2. an informative observation can empty the old support;
3. one candidate batch may not refill the requested support;
4. the current LLM set-overlap audit cannot certify replay stability.

The exact zero-retry interface is closed on these cases. No threshold, prompt, model,
or endpoint is changed post hoc. A distinct follow-up may implement the literature's
pre-specified repeated generate-filter cycle on fresh cases and use deterministic
matching diagnostics rather than this overlap judge. No policy run is authorized from
this gate alone.

## Cost

- 14 requests: 6 GPT-5.4 generation and 8 GPT-5.4 Mini filtering/audit;
- 6,215 prompt and 5,577 completion tokens;
- zero reasoning tokens;
- `$0.04314350`;
- conservative project-ledger remainder: `$21.78535522`.
