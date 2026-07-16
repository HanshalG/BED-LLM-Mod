# COPEx L3 Fixed-Plan Ranking Fidelity

Exploratory posthoc replay of the completed L3 artifact. No LLM calls, policy reruns, or new trajectories were used. The primary L3 intersection gate remains failed.

## Replay

- Input run: `nonmyopic-copex-strategy-l3-confirmation-recovery1-20260716`.
- Replayed steps: `900` / `900`; posterior and chosen-root assertions passed.
- Decision cells / candidates: `900` / `3600`.
- Selected root kinds: `{'midpoint_ranks': 65, 'toward_mean': 150, 'toward_rank': 419, 'vector': 266}`.

Each candidate follows its entire original macro plan from the recorded pre-decision posterior, using the completed trial's same future Gaussian innovations. This tests the finite-horizon score's ranking; it is not receding-horizon policy regret.

## Trial-Clustered Metrics

| Metric | Mean of trial means | Trial-bootstrap 95% CI | Trials / cells |
| --- | ---: | --- | ---: |
| Spearman: score vs entropy drop | 0.653551 | [0.601831, 0.694439] | 30 / 836 |
| Spearman: score vs truth log-probability delta | 0.333448 | [0.253719, 0.411750] | 30 / 295 |
| Predicted top-1 entropy accuracy | 0.626667 | [0.561111, 0.686667] | 30 / 900 |
| Predicted top-1 truth-log-probability accuracy | 0.504444 | [0.428889, 0.577778] | 30 / 900 |
| Predicted top-1 entropy regret | 0.061404 | [0.046908, 0.076942] | 30 / 900 |
| Predicted top-1 truth-log-probability regret | 0.082477 | [0.062723, 0.105558] | 30 / 900 |
| Score margin | 0.021904 | [0.017637, 0.026246] | 30 / 900 |
| Within-cell score standard deviation | 0.032332 | [0.027358, 0.037443] | 30 / 900 |

## Pooled Candidate Context

- Score vs fixed-plan entropy drop Spearman: `0.909100`.
- Score vs fixed-plan truth-log-probability delta Spearman: `0.728201`.

Positive correlations indicate ranking alignment. These descriptive mechanism metrics neither rescue nor revise the preregistered primary result.
