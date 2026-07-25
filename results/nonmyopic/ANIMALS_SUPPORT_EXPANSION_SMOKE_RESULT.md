# Animals Support-Expansion Smoke Result

Date: 2026-07-25

## Outcome

The frozen two-state serving/mechanics smoke passed every gate. It used
`google/gemma-4-26b-a4b-it` without thinking and completed 971 OpenRouter
requests with zero retries, reasoning tokens, or forced exits for
`$0.02914171`.

Both states retained three root candidates and had nonconstant expected
regenerated-support-size scores. Support expansion selected a different root
from immediate EIG on both states. Its model-averaged target-coverage score
was `.830256` versus `.619487` for immediate EIG, a paired gain of
`+.210769`.

Realized target coverage was `1.0` for both selectors on both smoke states, so
the smoke is not efficacy evidence. This saturation is permitted by the
mechanics-only smoke. The frozen development gate separately requires
immediate-EIG realized coverage to lie between `.05` and `.95`.

The production branch path was exercised: generated hypotheses were
structurally cleaned, animal-name validated, filtered against the complete
branch history, merged with surviving prior hypotheses, and retried only under
the existing minimum-support rule. Current-support EIG was exact, while branch
updates were not placed in the deterministic bypass mode.

## Artifacts

- Public result:
  `results/nonmyopic/animals_support_expansion_smoke/animals-support-expansion-smoke-20260725T211359Z/SERVING_SMOKE.json`
- Public SHA-256:
  `30c775f445112b545ee6eef616d5a0c274d4f368c12bc3e59fd05a8687d19ba5`
- Private raw SHA-256:
  `e71edabc19215e93097b5eefe722037853d5b3952b8fd5fd901ae05e138b958a`
- Frozen implementation commit: `1d5d39d`

The passing smoke authorizes exactly the preregistered 20-state development
run. The 60-target holdout remains sealed and is rejected by the runner.
