# LLM Strategy-Prior Result

## Conclusion

The preregistered claim is **not established**. The project produced a working
LLM-primary non-myopic BED architecture and encouraging proposal-quality evidence, but
StrategyEIG did not pass the required intersection gate against both shared myopic
scoring and grammar-matched random plans.

The clean continuous-location run gives a useful partial result: LLM StrategyEIG has
the lowest mean final entropy and lowest mean entropy AUC of all five arms, beats the
equal-compute width and discretized-d2 controls with positive final-entropy intervals,
and beats fixed-budget grid d2 across a 4/8/16 resolution sensitivity. However, its
final advantage over random plans is too unstable for the registered interval, and
shared d1 wins 24 of 30 final pairs. Thus the evidence supports **LLM proposal quality
in this plan grammar**, not a robust benefit from non-myopic plan scoring.

A subsequent zero-cost, explicitly posthoc ranking-fidelity replay narrows the
mechanistic diagnosis. The analytic scorer ranks the four candidates strongly by
their realized **fixed-plan** entropy drops, but that score fidelity does not turn
into a confirmed advantage for the live root-only receding-horizon policy. This is
useful evidence about where the method fails; it does not revise the failed primary
gate.

## What Was Built

- A strict compact natural-language strategy grammar with programmatic executors for
  discrete Rock Diagnosis and continuous constrained location.
- Exact branch enumeration for finite Rock strategies.
- An independent implementation of the COPEx `Location_budgeted` task equations,
  with exact finite-particle analytic likelihood/posterior updates and CRN Monte Carlo
  EIG estimation.
- Paired StrategyEIG, shared-d1, equal-compute width, grammar-matched random-plan, and
  equal-compute discretized-d2 controls.
- Verbatim strategy/request traces, fail-closed parsers, bounded feedback repair,
  scorer-unit accounting, and zero-LLM grid-resolution sensitivity.

All final mechanics passed: every action obeyed `[0,1]^2` and the L-infinity step-0.1
constraint; StrategyEIG/shared-d1 cells were shared when states coincided; width and
grid-d2 scorer units matched StrategyEIG; rollout scoring made zero LLM calls.

## Rock L1

The Rock L1 endpoint was never produced. The first formal invocation failed the
move/check root-mix interface after 316 requests. Its sole prompt-only retry accepted
857 cells, then failed closed on an illegal strategy branch. No arm metric was read
from either invocation, and the registered branch closed Rock rather than silently
substituting actions or relaxing legality.

This is itself a boundary result: Gemma 4 26B can generate many sensible compact Rock
plans, but the flexible ordered-rule language was not reliable enough for a
30-trial confirmatory run under strict all-branch legality.

## Continuous L3

Formal clean run: seed 31003; 30 paired trials; 30 rounds; 64 particles plus truth;
K=4; horizon 4; 64 CRN rollouts; non-thinking Gemma 4 26B. The completed run used
1,373 requests, 498,924 prompt tokens, 435,889 completion tokens, zero reasoning tokens,
and `$0.21004985`.

| Arm | Final entropy | Final RMSE | Mean scorer units / decision |
| --- | ---: | ---: | ---: |
| StrategyEIG | **0.000025** | 0.000000 | 972.8 |
| shared d1 | 0.000034 | 0.000000 | 256.0 |
| equal-compute width | 0.000143 | 0.000000 | 972.8 |
| random strategies | 0.001541 | 0.000002 | 972.8 |
| matched-budget grid d2 (resolution 8) | 0.000175 | 0.000000 | 972.8 |

Positive gain favors StrategyEIG.

| Comparison | Final entropy gain | 95% paired bootstrap CI | W / T / L |
| --- | ---: | --- | --- |
| StrategyEIG - shared d1 | +0.000010 | [-0.000000, +0.000029] | 4 / 2 / 24 |
| StrategyEIG - width | +0.000118 | [+0.000000, +0.000355] | 29 / 0 / 1 |
| StrategyEIG - random strategies | +0.001516 | [-0.000000, +0.004548] | 20 / 0 / 10 |
| StrategyEIG - grid d2 | +0.000150 | [+0.000001, +0.000401] | 28 / 0 / 2 |

The intersection gate fails on shared d1 and random strategies. Values displayed as
`-0.000000` or `+0.000000` are tiny bootstrap endpoints on opposite sides of zero, not
rounded evidence of a positive gate.

## Trajectory View

StrategyEIG has the best mean entropy AUC (`0.486403`) versus shared d1 (`0.493738`),
width (`0.511817`), random (`0.516091`), and grid d2 (`0.537106`). These are secondary,
descriptive endpoints. Pairwise AUC wins are only 12/30 versus shared d1, 16/30 versus
width, and 18/30 versus both random and grid d2.

All arms nearly saturate the task by 30 rounds, making the registered final endpoint
very small and heavy-tailed. Shared d1 is better on most final pairs even though a few
large StrategyEIG gains make its mean entropy lower. This is not the pattern expected
if horizon-4 scoring were reliably improving the first action.

Winning LLM roots are dominated by local exploitation: `toward_rank` (419/900),
direct vectors (266), `toward_mean` (150), and midpoint probes (65). Common plan names
include `local_refinement`, `centroid_convergence`, and
`local_refinement_cluster`. The LLM learned a coherent spatial prior, but that prior
mostly chooses locally sensible movement; deeper rollout scoring does not consistently
improve on scoring those same roots myopically.

## Grid Sensitivity

The sensitivity makes zero LLM calls and holds the evaluated d2 sequence budget at
eight while increasing the full enumerable tree.

| Resolution | Full d2 sequences | Evaluated | Grid final entropy | Strategy gain | 95% paired CI |
| ---: | ---: | ---: | ---: | ---: | --- |
| 4 | 16 | 8 | 0.001422 | +0.001397 | [+0.000001, +0.003619] |
| 8 | 64 | 8 | 0.000175 | +0.000150 | [+0.000001, +0.000400] |
| 16 | 256 | 8 | 0.004803 | +0.004778 | [+0.000029, +0.012445] |

This is the strongest positive evidence: at fixed search budget, the LLM plan prior
covers useful continuous geometry better than a resolution-dependent grid subset.
It remains a proposal-prior result because shared d1 shows that horizon-4 scoring is
not robustly load-bearing.

## Posthoc Ranking Fidelity

The registered posthoc diagnostic replays only the completed L3 histories: 900
StrategyEIG decision cells and 3,600 already-generated candidate plans. It makes zero
LLM calls and zero policy reruns. From every logged pre-decision posterior, it executes
each candidate's complete macro plan under the completed trial's true source and same
future Gaussian innovations. This is a fixed-plan score-fidelity test, not a claim
about deployed receding-horizon policy regret.

The score is meaningfully aligned with the simulator's realized fixed-plan information
gain: the mean trial-level within-cell Spearman score/entropy-drop correlation is
`0.653551` (trial-bootstrap 95% CI `[0.601831, 0.694439]`), and the score's top candidate
is the realized entropy-drop top candidate in `0.626667` of cells. The analogous
score/truth-log-posterior correlation is weaker, `0.333448` (`[0.253719, 0.411750]`),
with many late-round tied truth-posterior outcomes. Candidate score margins are small
(mean `0.021904`, versus within-cell score sd `0.032332`), and top-score selection still
leaves mean fixed-plan entropy regret `0.061404` nats.

This rules out a gross simulator-versus-realized-ranking mismatch as the dominant
explanation. The remaining gap is controller-level: a faithful fixed-plan score is
used only to choose the first action, after which the policy regenerates and replans;
the resulting roots are not consistently better than shared-d1 roots or grammar-matched
random-plan roots on the preregistered paired endpoint.

## Interpretation

The simple simulator-ranking mismatch is not the dominant bottleneck: the posthoc
fixed-plan replay finds strong score/entropy ordering under the same analytic model.
Instead, the unresolved controller-level links are:

1. The low score margins and nonzero counterfactual top-one regret show that the
   candidate choice still has material local uncertainty, even though the average
   rank signal is positive.
2. Receding regeneration plus posterior-relative macros makes strong plans locally
   adaptive even under d1; much of the LLM prior's value is already present in the root.
3. The 30-round task saturates, weakening final-entropy discrimination.
4. Gemma's generated plan family concentrates on exploitation, reducing the diversity
   of genuinely delayed-information strategies for horizon scoring to exploit.

A future project should preregister an earlier/AUC endpoint on a deliberately delayed-
information continuous task, test whether fixed-plan value differences improve deployed
root selection before scaling trials, and use a plan grammar that exposes meaningful
exploratory waypoints without brittle branch rules. Those are new experiments, not
post-hoc repairs to this result.

## Cost and Reproduction

Total live strategy-prior phase spend was `$0.78649365`, including interface smokes
and failed-closed invocations. The final cumulative OpenRouter tracker is
`$18.57112984` of `$40` authorized.

Primary artifacts:

- `results/nonmyopic/rock_strategy_l0_smoke/20260716/REPORT.json`
- `results/nonmyopic/rock_strategy_l1_confirmation/20260716/L1_FAILURE.json`
- `results/nonmyopic/rock_strategy_l1_confirmation_retry1/20260716/L1_FAILURE.json`
- `results/nonmyopic/copex_strategy_l3_confirmation_recovery1/20260716/L3.json`
- `results/nonmyopic/copex_strategy_l3_confirmation_recovery1/20260716/L3.md`
- `results/nonmyopic/copex_strategy_l3_grid_sensitivity/20260716/GRID_SENSITIVITY.json`
- `results/nonmyopic/copex_strategy_l3_grid_sensitivity/20260716/GRID_SENSITIVITY.md`
- `results/nonmyopic/COPEX_STRATEGY_L3_RANKING_FIDELITY_PREREGISTRATION.md`
- `results/nonmyopic/copex_strategy_l3_ranking_fidelity/20260716/REPORT.json`
- `results/nonmyopic/copex_strategy_l3_ranking_fidelity/20260716/REPORT.md`

Final repository verification: `664 passed, 1 skipped`; experiments-ledger validation
passes with no active rows and all complete artifacts present.
