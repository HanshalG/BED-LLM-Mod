# ClinDiag Staged Open-World Generator Gate Result

Date: 2026-07-24

Status: **all frozen development gates passed; branch-mechanics development is
authorized, but no policy or holdout evaluation is yet authorized.**

## Protocol

The official Apache-2.0 ClinDiag-Benchmark archive was pinned at commit
`f9b5c181e8d120d6a244accba9422ffb89d1f319` and SHA-256
`a9ea339fc3a6ded91f5c1769589c5c374dd32c266072511260d4102e544539c1`.
Before any response, seed `24289` fixed disjoint four-case serving, 20-case
development, and 60-case holdout splits. The holdout remains untouched.

Eligibility required every native evidence stage, a final diagnosis of at most 12
normalized words, and no exact normalized diagnosis or first-four-token diagnosis
prefix in any evidence. The resulting development split contained 10 challenging and
10 rare cases.

Non-reasoning GPT-5.4 generated 12 free-form diagnoses from:

1. the one-sentence initial presentation;
2. the initial presentation plus complete medical history, physical examination, and
   diagnostic tests.

The true diagnosis was absent from both prompts. Non-reasoning GPT-5.4 Mini received
it only after generation for strict semantic-equivalence measurement at threshold
`0.80`.

## Frozen Gate

| Criterion | Required | Observed | Pass |
|---|---:|---:|---:|
| Complete cases | 20 | 20 | yes |
| Initial coverage | at most 8 | 6 | yes |
| Full-evidence coverage | at least 16 | 17 | yes |
| Initial omissions recovered | at least 10 | 12 | yes |
| Recovery fraction | at least 0.70 | 0.8571 | yes |
| Mean semantic-match gain | at least +0.30 | +0.4125 | yes |
| Challenging full coverage | at least 8/10 | 10/10 | yes |
| Rare full coverage | at least 7/10 | 7/10 | yes |

All criteria passed. Complete evidence recovered 12 of 14 initially omitted diagnoses
and raised support coverage from 6/20 to 17/20 without prior saturation.

## Recoveries And Misses

Exact or clinically equivalent recoveries included Sotos syndrome, congenital
hypothyroidism due to thyroid hypoplasia, periventricular nodular heterotopia,
nail-patella syndrome, wild-type ATTR amyloidosis, insulinoma, rectal duplication,
cutaneous polyarteritis nodosa, catastrophic antiphospholipid syndrome, and familial
cold urticaria.

The three full-evidence misses were all rare-tail distinctions:

- RNF13-related severe early-onset epileptic encephalopathy remained `0.05`;
- Micro syndrome fell from `0.85` to `0.78` under the strict subtype rule;
- keratocystic odontogenic tumor remained `0.00`, with odontogenic keratocyst judged
  related but not strictly equivalent.

## Integrity And Cost

- Serving smoke: exactly 10 requests, all eight generated supports size 12, zero
  retries/reasoning/runtime failures, `$0.03608825`.
- Development: exactly 60 requests, 40,881 prompt tokens, 8,106 completion tokens,
  zero retries/reasoning/forced exits/runtime failures, `$0.17443725`.
- Combined line cost through this gate: `$0.21052550`.
- Project-ledger spend after the gate: `$45.02778097`, leaving `$25.35702172` under
  the conservative fail-closed ceiling.
- No answer option, case title, final diagnosis, semantic score, or holdout record
  entered a generation prompt.

## Consequence

This is the first external open-world gate in the project to demonstrate reliable
observation-driven support recovery. It establishes the missing upstream mechanism:
an LLM-generated belief support can be non-saturated initially and substantially
improve when native evidence arrives.

It does not establish that a target-blind policy can predict which evidence will cause
that improvement, that evidence order matters, or that non-myopic selection beats a
myopic control. The next authorized experiment is therefore a fresh development-only
branch-opportunity gate over ClinDiag's native evidence categories. The 60-case holdout
and any policy claim remain sealed.
