# Number Game Multi-Draw Prior Development Result

Date: 2026-07-28

Protocol:
`NUMBER_GAME_MULTIDRAW_PRIOR_DEVELOPMENT_PREREGISTRATION.md`

Run:
`results/nonmyopic/number_game_multidraw_prior_development/number-game-multidraw-prior-development-20260728T140000Z`

## Verdict

**Development null. The three-draw proposal prior is closed.**

All 16 target-blind Gemini calls completed with valid executable supports, no
retry, reasoning, forced exit, or budget violation. Seven of eight pooled
supports exceeded the frozen 32-extension breadth threshold; one contained 31.

The additional draws did not improve root selection:

| Baseline | Multi-draw Brier | Baseline Brier | Relative gain | Strict wins | Whole-tree difference CI |
|---|---:|---:|---:|---:|---:|
| original one-draw predictive risk | 0.18739 | 0.18632 | -0.58% | 1/8 | [-0.00266, 0.00583] |
| PTS | 0.18739 | 0.19176 | 2.28% | 5/8 | [-0.01507, 0.00568] |
| myopic EIG | 0.18739 | 0.22094 | 15.18% | 7/8 | [-0.04812, -0.01826] |

The three-draw policy changed only three roots. One change improved the exposed
endpoint, one was nearly neutral, and one substantially worsened it. All frozen
gates against the original policy and PTS fail. The robust myopic advantage
persists.

More unweighted prior draws are therefore not justified. They add hypotheses
but do not calibrate their transfer relevance. The next test keeps the
successful one-draw method unchanged and increases independent-tree power for
the small observed PTS difference.

## Accounting

- Accepted requests / HTTP attempts: `16 / 16`
- Retries / reasoning tokens / forced exits: `0 / 0 / 0`
- Cost: `$0.0361394`
- Result SHA-256:
  `349112658e11066e02c207b56407a02eab1c8895bd1ac2a82f5a14eadbec55f0`
- Extra-prior artifact SHA-256:
  `710d3ce051c68d6ad6e1ef263a1e88dd3f8f469bea692a543d9a24526b9dfd31`
- Private raw-response SHA-256:
  `fe4aa7bc11b37c5efbb02ece85d12340f4a23164840589005a3020a6b467b181`
