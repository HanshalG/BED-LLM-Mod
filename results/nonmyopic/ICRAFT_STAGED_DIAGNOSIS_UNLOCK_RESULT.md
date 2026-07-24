# iCRAFT Staged Open-World Diagnosis Unlock Result

Date: 2026-07-24

Status: **development gate failed; the 60-case holdout and planner remain
unauthorized.**

## Protocol

This derived task is distinct from the closed official FactSelect profile protocol.
Twenty fixed development cases were selected by seed `24288`, excluding all 13 iCRAFT
IDs that had previously received model calls. The 60-case holdout was not touched.

Non-thinking Gemma 4 26B generated:

1. eight free-form diagnoses from only the first two released atomic facts;
2. eight new diagnoses after every remaining released fact was revealed as a guaranteed
   workup packet.

The multiple-choice options and true diagnosis were absent from both prompts.
Non-reasoning GPT-5.4 Mini saw the true diagnosis only after generation and measured
semantic equivalence at threshold `0.80`.

## Frozen Gate

| Criterion | Required | Observed | Pass |
|---|---:|---:|---:|
| Complete tasks | 20 | 20 | yes |
| Initial coverage | at most 10 | 5 | yes |
| Workup-generated coverage | at least 14 | 11 | **no** |
| Initial omissions recovered | at least 8 | 6 | **no** |
| Omission recovery fraction | at least 0.60 | 0.40 | **no** |
| Mean semantic-match gain | at least 0.25 | 0.3375 | yes |

The workup more than doubled exact semantic coverage from 5/20 to 11/20 and recovered
6/15 initially omitted diagnoses. This is stronger structured-unlock evidence than the
Animals and Paprika gates, but it misses every frozen reliability threshold.

## Cases

Exact recoveries included:

- Toxicodendron dermatitis: `0.20 -> 0.95`;
- pellagra: `0.00 -> 1.00`;
- lichen planus: `0.00 -> 1.00`;
- IgA-mediated small-vessel vasculitis: `0.00 -> 1.00`;
- tuberous sclerosis complex: `0.00 -> 1.00`;
- poison ivy allergic contact dermatitis: `0.74 -> 0.97`.

Large subthreshold improvements included levamisole-induced ANCA vasculitis
`0.10 -> 0.70` and Hodgkin lymphoma `0.00 -> 0.70`. Full evidence still failed to
recover several tail diagnoses, including Grover disease, cutaneous Langerhans cell
histiocytosis, acrodermatitis enteropathica, and
Conradi-Hünermann-Happle syndrome.

## Integrity And Cost

- Serving smoke: exactly 10 requests, all diagnosis supports size eight, zero
  reasoning, `$0.00123355`.
- Development: exactly 60 requests, 15,661 prompt tokens, 5,664 completion tokens,
  zero reasoning, `$0.01746645`.
- Project-ledger spend: `$44.81725547`.
- Live OpenRouter balance checked after the gate: `$25.56754722` remaining
  (`$80.00` credited, `$54.43245278` used account-wide).
- No holdout case, answer option, or true diagnosis entered a generation prompt.

## Consequence

The result isolates a promising but incomplete mechanism: guaranteed structured
evidence causes substantial open-world hypothesis recovery without prior saturation.
The remaining bottleneck is full-evidence diagnosis generation on rare labels, not
answer availability.

Per the frozen stop rule, this exact staged-iCRAFT/Gemma line gets no workup candidates,
likelihood elicitation, target-blind ranker, policy run, model swap, support-width tune,
or holdout evaluation. A future task must preserve the same partial-to-structured
evidence transition while supplying a hypothesis generator that is independently
qualified on full evidence before planning.
