# iCRAFT Staged Open-World Diagnosis Unlock

Date: 2026-07-24

Status: preregistered development mechanism gate. This is a distinct derived
staged-evidence task, not a retry of the closed official FactSelect profile protocol.

## Motivation

The official iCRAFT FactSelect gate failed because only 14/48 open-ended questions were
answerable from the static records. Here every released atomic fact is guaranteed to be
available, but evidence is deliberately staged:

1. the initial open-world differential sees only the first two released facts;
2. a diagnostic workup reveals every remaining fact;
3. the LLM regenerates the differential from the complete staged evidence.

Answer options and the benchmark label are hidden from both generation stages. The
target appears only in a final semantic-equivalence measurement. This tests the
upstream prerequisite for an irreducible-LLM non-myopic environment: partial evidence
must omit the true semantic hypothesis, and a guaranteed workup must recover it.

## Frozen Split

The 13 source IDs that previously received iCRAFT model calls are excluded:
`2, 23, 40, 60, 62, 64, 96, 99, 100, 117, 125, 132, 137`.

NumPy seed `24288` shuffled the remaining 127 IDs before any new response:

- development (20): `8, 13, 14, 29, 37, 38, 46, 55, 66, 81, 84, 85, 88, 98,
  105, 109, 110, 114, 121, 138`;
- untouched holdout (60): recorded in
  `scripts/icraft_staged_diagnosis_unlock.py`;
- unused reserve: 47.

The holdout cannot be used unless this development gate passes and a target-blind
planner is frozen.

## Models And Endpoint

- Non-thinking `google/gemma-4-26b-a4b-it`.
- Eight initial diagnosis names and eight workup-conditioned diagnosis names.
- Non-reasoning `openai/gpt-5.4-mini` semantic measurement.
- Semantic coverage threshold `0.80`, requiring the same diagnosis or a standard
  clinical synonym; broad categories, symptoms, related diseases, and alternative
  subtypes do not count.
- Exactly 10 physical requests must pass a serving smoke first.
- Development projected cost `$0.25`, hard run cap `$1.00`.

## Frozen Pass Gate

All conditions must hold:

1. all 20 development tasks complete;
2. initial support covers at most 10/20 truths;
3. workup-generated support covers at least 14/20 truths;
4. at least eight initial omissions are recovered;
5. at least 60% of initial omissions are recovered;
6. mean workup-minus-initial best-match score is at least `0.25`.

Failure closes this staged iCRAFT line before candidate workups, likelihood elicitation,
ranking, or policy evaluation. Passage authorizes only a target-blind workup-selection
development phase, not use of the untouched holdout.
