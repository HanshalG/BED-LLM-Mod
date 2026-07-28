# Number Game Generator-Aware BED Development Result

Date: 2026-07-28

Protocol:
`results/nonmyopic/NUMBER_GAME_GENERATOR_AWARE_BED_PREREGISTRATION.md`

Run:
`results/nonmyopic/number_game_generator_aware_bed/number-game-generator-aware-bed-20260728T120000Z`

## Verdict

**Clean mechanics null. The max-future-EIG objective is closed.**

The run made exactly 17 accepted Gemini 2.5 Flash calls with no retry,
reasoning token, forced exit, parse failure, or budget violation. The initial
support retained 23 unique executable extensions, and every branch retained at
least 13. The generator-aware policy chose root 72 rather than the shared
myopic and fixed-support-depth-two root 48.

That difference was not meaningful. Rich regenerated supports almost always
contained a near-balanced second query, so maximum second-step EIG saturated
near one bit in every branch. Generator-aware scores were:

- root 72: `1.384749` nats;
- root 48: `1.384227` nats;
- difference: `0.000521` nats, below the frozen `0.01`-nat gate.

The selected root survived only `16/23` leave-one-out particle replays
(`69.6%`), below the frozen 75% gate.

## Published-Target Endpoint

| Policy | Root | Mean Brier | Best-rule Hamming | Exact-extension coverage |
|---|---:|---:|---:|---:|
| generator-aware depth two | 72 | 0.18587 | 0.11799 | 4/12 |
| myopic EIG | 48 | 0.16624 | 0.10231 | 5/12 |
| fixed-support depth two | 48 | 0.16624 | 0.10231 | 5/12 |
| deterministic random | 34 | 0.17300 | 0.11799 | 4/12 |

Generator-aware depth two worsened Brier by `11.8%`, worsened best-rule
Hamming error by `15.3%`, and lost one exact target extension versus myopic
EIG. The preregistered development signal therefore fails.

## Diagnosis

Proposal validity alone is not the missing utility. Root 72 had the highest
expected valid-unique proposal rate (`89.5%`) but generalized worse than root
48. The failure is instead in the terminal value function: maximizing the best
next hypothesis-identity EIG asks only whether a regenerated set can be split
in half. It does not ask whether that set retains the simulated truth or makes
accurate predictions over the Number Game domain.

A zero-call diagnostic on the already-open tree shows the principled successor.
For each current particle treated as truth, execute the regenerated branch,
choose its greedy second query, and score the resulting posterior predictive
Brier loss against that simulated truth. This empirical Bayes-risk objective
selects root 34. On the current-particle development distribution, root 34 has:

- mean Brier `0.1858` versus myopic root 48 at `0.2116` (`12.2%` lower);
- best-rule Hamming `0.0732` versus `0.0943` (`22.4%` lower);
- truth-extension retention `69.6%` versus `65.2%`.

Those are posthoc development diagnostics, not evidence. A successor must be
frozen before testing root 34 on fresh independently generated target concepts,
and a positive target result must still transfer to independent proposal trees.

## Accounting

- Accepted requests / HTTP attempts: `17 / 17`
- Retries / reasoning tokens / forced exits: `0 / 0 / 0`
- Cost: `$0.039824`
- Model artifact SHA-256:
  `bf45eb9f5dd8da0289d834671952b53f9dff8fc7144190fe1ecfab50045bcecb`
- Private raw-response SHA-256:
  `f114528a99ee10dcdcb45f739937b48562d559f244c726798b4e002feb0f0a36`
