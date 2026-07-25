# Zendo Final-Readiness Belief Smoke Result

Date: 2026-07-25

Status: **failed closed before scoring or hidden-endpoint evaluation**.

Preregistration:
`results/nonmyopic/ZENDO_FINAL_READINESS_BELIEF_SMOKE_PREREGISTRATION.md`

Run:
`zendo-final-readiness-smoke-20260725T072302Z`

## Outcome

The initial executable support and all eight outcome-conditioned refresh responses
were returned as complete JSON objects. Seven refreshed populations contained 12
unique validated ASTs. The `root_2_yes` population contained 12 rows but only 11
unique ASTs: one executable rule was duplicated.

The frozen parser therefore rejected the branch. The run stopped after nine
physical requests, before the final-readiness scorer was called. The official
`phi` truth function was never used to score a policy or calculate a realized
endpoint.

There was no deduplication, replacement, coercion, branch reissue, threshold
change, or rerun. This exact interface is closed.

## Serving diagnostics

- Physical requests / HTTP attempts: `9 / 9`
- Prompt / completion / reasoning tokens: `8,896 / 5,518 / 0`
- Retries / forced exits: `0 / 0`
- Cost: `$0.10501`
- Initial response present: yes
- Branch responses present: `8 / 8`
- Strict JSON top-level shape valid: `8 / 8`
- Branches with 12 unique ASTs: `7 / 8`
- Scorer response: absent by design after failure
- Hidden policy endpoint: not evaluated

The failure is a support-validity failure, not evidence for or against the
final-readiness ranking hypothesis. It does show that requiring a fixed-size
unique particle set directly from an unconstrained generator remains brittle even
for GPT-5.4 non-reasoning.

## Reproducibility

- Public failure artifact:
  `results/nonmyopic/zendo_final_readiness_belief_smoke/RESULT.json`
- Private raw-response SHA-256:
  `eb7e6ec1916f00a994f961575b6c32e5d8f2ead6ee895ad34683cb16651443dc`
- Run failure artifact SHA-256:
  `640d965accac0070a70f4939a15376678858f73f2b3c13094549acd4edb576d7`

Raw model responses remain untracked.
