# Number Game Depth-Three Development V2 Result

Date: 2026-07-28

Protocol:
`NUMBER_GAME_DEPTH_THREE_DEVELOPMENT_V2_PREREGISTRATION.md`

Run:
`results/nonmyopic/number_game_depth_three_development_v2/number-game-depth-three-development-v2-20260728T193000Z`

## Verdict

**Development null: strong Brier signal, failed mechanics and coverage gates.**

Depth three changes 7/8 first roots and improves endpoint Brier over depth two
by `12.32%`, with a wholly negative whole-tree interval and 6/8 wins. It does
not promote because three twice-conditioned branch cells retain zero valid
hypotheses, eight fall below the frozen minimum of four, and mean
truth-extension coverage falls by `1.71` percentage points.

No empty branch was regenerated, tree removed, or threshold changed. This
implementation of a third planning step is closed without confirmation.

## Depth Three Versus Controls

| Baseline | Depth-three Brier | Baseline Brier | Relative gain | Wins | Whole-tree 95% difference CI |
|---|---:|---:|---:|---:|---:|
| depth-two predictive risk | 0.18951 | 0.21614 | 12.32% | 6/8 | [-0.05267, -0.00658] |
| myopic EIG | 0.18951 | 0.20708 | 8.49% | 5/8 | [-0.03005, -0.00572] |
| fixed-support depth three | 0.18951 | 0.22328 | 15.13% | 5/8 | [-0.05873, -0.01103] |
| exact uniform random | 0.18951 | 0.21660 | 12.51% | 8/8 | [-0.03526, -0.01997] |
| seeded PTS | 0.18951 | 0.22043 | 14.03% | 7/8 | [-0.05212, -0.01155] |

Versus depth two, best-rule Hamming improves by `8.20%` (`0.15040` versus
`0.16383`), but its interval `[-0.03949, 0.01075]` crosses zero. Coverage is
the decisive scientific failure.

## Mechanics Failure

All initial supports contain at least 21 valid rules, first-step branches at
least 10, target supports at least 21, and every tree has at least eight target
extensions novel to its initial support. Second-step minimum is zero.

Three of 256 twice-conditioned cells are empty:

- seed `27600`, history `5:NO, 27:YES`: all 24 rules contradicted the history;
- seed `27606`, history `87:YES, 3:NO`: all 24 contradicted the history; and
- seed `27607`, history `19:YES, 0:YES`: all 24 used invalid grammar.

Five other cells retain only one to three unique valid rules.

## Replay Provenance

All 400 model responses completed normally. The original process then hit a
local `KeyError` because the shared aggregator hard-coded the two-query
candidate name. A deterministic zero-call replay fed the frozen raw responses
through the same parser and evaluator after parameterizing that candidate
name. `TREES.json` and `RESULT.json` declare this replay.

## Accounting

- Successful responses / HTTP attempts: `400 / 400`
- Retries / provider-error retries: `0 / 0`
- Reasoning tokens / forced exits: `0 / 0`
- Cost: `$1.2877987`
- Result SHA-256:
  `b3eb13f98edd4235990a4084953855030db008db1e3185778ea2781e4583376a`
- Tree artifact SHA-256:
  `016ed9218e9f034984e0745c9a7cb62b4111901db821c065e20c38c5ce76b65f`
- Private raw-response SHA-256:
  `3cfaa68e16dcc869244dba106492bff3fa2536d32715bf3990cad09d2ddaa0ce`
- Run-log SHA-256:
  `2f6a62918133e2f02797b1b62248d49fbb4b62356cea4399becf6bc88866ee07`
