# ClinDiag BED-LLM Filtered-Retention Serving Gate

Date: 2026-07-24

Status: **preregistered before any serving call.**

## Rationale

The prior-retaining ClinDiag prompt was reproducible but too inert: exhaustive ordered
pairs produced zero validated truth unlocks. Rebuilding the whole differential from
scratch was responsive but unstable even at temperature zero.

This gate tests the distinct update mechanism used by BED-LLM
(https://arxiv.org/pdf/2508.21184): generate candidates jointly, filter new candidates
against the full history, filter old hypotheses against the newest observation, then
merge the survivors. The rejection threshold is frozen at `0.20`, matching BED-LLM
Appendix E. This is also consistent with selective particle revision in "Doing
Experiments and Revising Rules" rather than replacing every hypothesis.

## Frozen Cases And Update

Seed `24299` selected one unused challenging and one unused rare case from the static
fixed-slot eligible pool after excluding every previously used fixed-slot case:

- challenging `11388546`;
- rare `rare203`.

Each case receives one deterministic stored `lab_1` observation. The action label and
finding come from the archive. The hidden target is never supplied to generation or
filtering.

For each case:

1. generate a 12-diagnosis initial support;
2. retain old diagnoses whose estimated likelihood of `lab_1` is at least `0.20`;
3. generate 12 diverse candidates from initial presentation plus `lab_1`, without
   showing the old support;
4. retain new candidates only when every visible evidence item's likelihood is at
   least `0.20`;
5. merge retained old diagnoses followed by valid new diagnoses, dedupe, and cap at
   12;
6. independently replay the exact candidate-generation prompt and repeat steps 4-5;
7. audit truth coverage and semantic final-support overlap.

Likelihood means `p(observed evidence | diagnosis)`, not a posterior diagnosis
probability. The minimum item likelihood controls filtering.

## Models, Calls, And Cost

- generation: `openai/gpt-5.4`, reasoning disabled, temperature `0`;
- filtering and audit: `openai/gpt-5.4-mini`, reasoning disabled, temperature `0`;
- structured retries: zero;
- exact physical requests: `14`;
- OpenRouter run ceiling: `$0.50`;
- projected ledger reservation: `$0.15`;
- live balance before launch must be checked and the stricter live/ledger remainder
  used.

The 14 requests are two initial generations, two old-support filters, four exact-prompt
candidate generations, four full-history candidate filters, and two semantic audits.

## Frozen Pass Rule

All conditions must hold:

1. exactly 14 requests, zero reasoning, zero structured retries, and no runtime error;
2. both final merged supports and both replay supports contain exactly 12 diagnoses;
3. at least one old hypothesis is pruned in each case;
4. at least one filtered new hypothesis enters each original and replay support;
5. worse-direction semantic overlap between final and replay is at least `0.80` in
   both cases;
6. original-versus-replay truth-score gap is at most `0.05` in both cases;
7. no full hidden target occurs in visible source evidence.

Passing establishes only a usable belief-update interface. It authorizes a fresh,
truth-anchored structural opportunity screen before semantic-likelihood or policy
experiments. Failure closes this exact filtered-retention interface without threshold,
prompt, model, or replay tuning on these cases.
