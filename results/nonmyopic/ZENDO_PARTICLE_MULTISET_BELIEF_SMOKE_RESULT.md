# Zendo Particle-Multiset Belief Smoke Result

Date: 2026-07-25

Status: **passed every frozen mechanics and scientific gate**.

Run: `zendo-particle-multiset-smoke-20260725T072852Z`

Preregistration:
`results/nonmyopic/ZENDO_PARTICLE_MULTISET_BELIEF_SMOKE_PREREGISTRATION.md`

## Main result

The blinded final-readiness scorer selected root 4 while both exact one-step EIG
and fixed-support exact depth-two EIG selected root 1.

| Root | Immediate EIG | Fixed d2 | Readiness score | Realized weighted truth agreement |
|---:|---:|---:|---:|---:|
| 1 | 0.494632 | 0.989264 | 50 | 0.636044 |
| 2 | 0.389654 | 0.879235 | 43 | 0.614732 |
| 3 | 0.301887 | 0.796519 | 47 | 0.557119 |
| 4 | 0.178255 | 0.664568 | 56 | 0.732650 |

Thus the model-aware root:

- sacrificed `0.316377` nats of immediate EIG;
- improved realized posterior-weighted behavioral truth agreement by `0.096605`
  over myopic and fixed-support depth two;
- achieved readiness-score versus realized-endpoint Spearman `0.80`;
- selected the best of the four roots on the external endpoint.

The realized endpoint range was `0.175530`, so the result is not a saturation or
tie artifact.

## Mechanics

- Exact requests / HTTP attempts: `10 / 10`
- Retries / reasoning tokens / forced exits: `0 / 0 / 0`
- Prompt / completion tokens: `14,374 / 5,503`
- Cost: `$0.11848`
- Initial behavioral signatures: `12 / 12`
- Behaviorally distinct branch populations: `8 / 8`
- Unique ASTs in every 12-particle population: `12 / 12`
- All continuations: positive finite exact EIG
- Readiness scores: varying with a unique maximum
- Repairs, response replacements, or scientific reruns: `0`

The multiset allowance was not needed by the realized responses: every population
was unique. It nevertheless defined valid particle semantics prospectively and
prevented a duplicate sample from becoming an arbitrary parser failure.

## Mechanism reading

The selected root was the least myopically informative candidate, with predicted
positive-label probability `0.875`. Its actual negative branch plus exact
continuation concentrated `0.7521` posterior mass on one refreshed executable
rule whose behavioral agreement with hidden `mu` was `0.7598`. The myopic root's
actual branch spread most posterior mass equally across four weaker rules, giving
weighted agreement `0.6360`.

The generated population did not recover the exact hidden rule: maximum agreement
after the selected path was `0.7598`. The valid claim is therefore that the LLM
ranked which first action would induce a **better regenerated approximate belief
state**, not that it solved Zendo or identified `mu` exactly.

This is a one-task development smoke. It is strong enough to authorize a
separately preregistered fresh-rule confirmation, but it is not by itself a
population result or a paper headline.

## Reproducibility

- Public result:
  `results/nonmyopic/zendo_particle_multiset_belief_smoke/RESULT.json`
- Full run artifact SHA-256:
  `49624677554f895ed70ca36e6ac9cbf53b9c2cd4c1706c0a2d84e93ee05a457c`
- Private raw-response SHA-256:
  `96edeaf44190c16e756a67a3a7e2f0f352f96e3d2fa11cd29f308b6c0aeea0cd`
- Scene-pool SHA-256:
  `80df2912f62d4f8f553ee78ba91fe86742532aaeb6a03738b23ef3e0d8f10abc`
- Audit-bank SHA-256:
  `34baa977b1bfcbba82cb514e96488c7c1f1933bc1fc48b86c9e5f96e6433bc99`

Raw model responses remain untracked.
