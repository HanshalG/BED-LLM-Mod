# InfoQuest Max-Resolution Confirmation Mechanics Result

## Status

The preregistered fresh-record mechanics run completed exactly, but the
scientific gates failed. The max-resolution aggregation route is closed.

- run ID:
  `infoquest-max-resolution-mechanics-20260726T043809Z`;
- public artifact SHA-256:
  `078b645675347bc63cdf05ec04ae16495764d4c556131c188c35d7e04968e674`;
- private raw SHA-256:
  `996455fb27fd5fa5f64df5476aa8bc8ecf4e3c01f3e4452c40cbca2aef27b055`;
- private fixture SHA-256:
  `e2f144df92292eae221349c45af2cfd81804caf3e77f0354907a7c4029954b87`.

## Accounting

The run made exactly 115 physical requests and 115 HTTP attempts: five
GPT-5.4 root-bank calls, 50 Gemini-2.5-Flash hidden-world root answers, 50
GPT-5.4 information-need beliefs, and ten GPT-5.4 Mini checklist judgments.
All calls were non-reasoning at temperature zero. Every strict parser passed,
with zero retries, reasoning tokens, and forced exits. Cost was `$0.17626315`,
below the frozen `$0.60` cap.

All compiler responses were checkpointed before any checklist labels were
generated. The compiler never received the hidden target or checklist content.
No response was repaired or reissued, and no holdout record was accessed.

## Scientific Result

The opportunity and support checks passed: 38/50 cells had target-gain spread,
41/50 had max-score spread, and oracle gain was `.88` compared with `.34` for
the seeded-random control.

The frozen max-resolution score did not confirm. Its mean within-cell
score/target-gain Spearman correlation was `-.2022`, below the `.20` gate and
opposite in sign. It selected `.28` target bits per cell, versus `.34` for both
the same-belief linear score and seeded random. Pairwise results were 5/37/8
wins/ties/losses against linear and 7/34/9 against random. Only 2/10 fixtures
improved over linear, although 24/50 choices were target-optimal.

## Interpretation

The strong post hoc development result for max-resolution did not replicate on
fresh records. The failure is not explained by absent action opportunity,
transport errors, retries, reasoning, or target leakage: the LLM-generated
belief itself attached high direct-resolution probability to distinctions that
were negatively aligned with realized target information.

This result closes both the linear and max aggregation variants of the current
information-need compiler. It does not support a policy or non-myopic claim,
and it will not be repaired or rerun on these records.
