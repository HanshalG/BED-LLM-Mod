# ClinDiag Staged Open-World Generator Gate

Date frozen: 2026-07-24

Status: **preregistered before any ClinDiag model response.**

## Purpose

The staged iCRAFT diagnostic established genuine observation-driven hypothesis
recovery, but complete evidence recovered only 6/15 omissions. Before attempting
another planner, this gate asks whether a stronger open-world generator can reliably
recover a hidden diagnosis from a richer, native staged record.

ClinDiag-Benchmark is a distinct external task with 2,021 real-world common and rare
cases. Each case separately releases initial information, medical history, physical
examination, diagnostic tests, and final diagnosis. This supplies guaranteed staged
evidence without reducing the target to a small enumerated ontology.

## Frozen Data

- Official source: `geteff1/ClinDiag`, commit
  `f9b5c181e8d120d6a244accba9422ffb89d1f319`.
- Archive SHA-256:
  `a9ea339fc3a6ded91f5c1769589c5c374dd32c266072511260d4102e544539c1`.
- Apache-2.0 release.
- Eligibility requires all five JSON stages, a diagnosis of at most 12 normalized
  words, and no exact normalized diagnosis or first-four-token prefix in any evidence
  stage.
- The static audit found 1,347 eligible cases: 1,098 challenging and 249 rare.
- Seed `24289` fixes four serving-smoke cases, 20 development cases, and 60 untouched
  holdout cases. Development is 10 challenging and 10 rare; holdout is 30 and 30.
- Exact IDs are frozen in `scripts/clindiag_staged_generator_gate.py`.

The smoke, development, and holdout sets are disjoint. The holdout receives no model
call unless this development gate passes.

## Models And Supports

- Generator: `openai/gpt-5.4`, reasoning disabled.
- Semantic measurement: `openai/gpt-5.4-mini`, reasoning disabled.
- Initial support: 12 free-form diagnoses from only `initial_information`.
- Full support: 12 regenerated diagnoses after medical history, physical examination,
  and all diagnostic tests are revealed.
- The true diagnosis is absent from both generation prompts and enters only the
  post-generation semantic measurement.
- Coverage threshold: `0.80`, requiring the same diagnosis or a standard clinical
  synonym rather than a related disease, broad parent, or alternate subtype.

## Serving Smoke

Before development, run exactly 10 physical requests:

- four initial generations;
- four full-evidence generations;
- two semantic measurements.

Pass requires all supports to parse to 12 unique diagnoses, exactly 10 requests, zero
reasoning tokens, no runtime failure, and no retry. Smoke endpoints are interface-only.

## Frozen Development Gate

All criteria must pass on the 20 development cases:

| Criterion | Requirement |
|---|---:|
| Complete cases | 20 |
| Initial coverage | at most 8/20 |
| Full-evidence coverage | at least 16/20 |
| Initially omitted diagnoses recovered | at least 10 |
| Recovery fraction among omissions | at least 0.70 |
| Mean full-minus-initial semantic score | at least +0.30 |
| Challenging-subset full coverage | at least 8/10 |
| Rare-subset full coverage | at least 7/10 |

Failure closes this exact ClinDiag/GPT-5.4 generator line before candidate actions,
likelihood elicitation, branch simulation, ranking, policy evaluation, or holdout use.
Pass authorizes a separate target-blind branch-mechanics design; it does not itself
authorize a policy claim.

## Cost And Integrity

- Project ledger spend before this line: `$44.81725547`.
- Live OpenRouter balance before this line: `$25.56754722`.
- Local fail-closed ceiling: `$70.38480270`; per-run cap: `$1.00`.
- Serving and development artifacts must report model-specific requests, tokens,
  reasoning tokens, and cost.
- No diagnosis option, case title, final diagnosis, semantic score, or holdout record
  may enter a generator prompt.
