# RegretBench Branch-Draw Fidelity Protocol

Date frozen: 2026-08-07, before any RegretBench policy response or endpoint.

## Purpose

The frozen dynamic policy averages two answer-conditioned LLM branch draws.
The existing draw-stability diagnostic reports whether their argmin roots
agree, but root agreement alone cannot distinguish sampling variance from a
shared ranking error. This audit measures whether averaging the two frozen
draws improves prediction of the realized dynamic-versus-refresh-myopic
advantage.

## Inputs

Run only on a complete 64-task RegretBench development or confirmation result
whose stored producer-independent `VERIFICATION.json` is `verified`, has no
mismatches, binds the exact `RESULT.json` SHA-256, and confirms that the result
matches replay. Both the original dynamic-depth-two and SMC dynamic-depth-two
result interfaces are eligible.

If the result is mechanically invalid, emit an unavailable diagnostic with no
fidelity metrics. Never interpret invalid endpoints.

## Frozen Task Set

Use exactly the tasks where the averaged dynamic policy and the
refresh-matched myopic-Brier policy selected different first roots. The task
set is determined by the frozen selections; no support, validity, stability,
or outcome-based subset may replace or filter it.

For each included task and branch draw `j`:

```text
predicted_advantage_j = risk_j(refresh_root) - risk_j(dynamic_root)
realized_advantage = Brier(refresh_policy) - Brier(dynamic_policy)
```

Positive values favor the averaged dynamic selection. The ensemble prediction
is the arithmetic mean of the two draw predictions. It must also equal the
same root contrast from the stored averaged conditioned-risk vector to within
`1e-12`.

## Metrics

Report for draw zero, draw one, and their average:

- Spearman correlation with realized advantage;
- paired task-bootstrap 95% interval and probability of positive correlation;
- sign accuracy, with exact-zero predictions or outcomes excluded;
- root-mean-square error and mean absolute error on the Brier-advantage scale.

Using one paired task bootstrap with 20,000 samples and seed `202608410000`,
also report the probability that the averaged Spearman correlation exceeds
draw zero, draw one, and both individual draws. Report the two draws' Pearson
correlation, mean absolute prediction gap, and root-mean-square prediction
gap. Degenerate bootstrap samples are omitted jointly from all correlation
contrasts and their retained count is reported.

## Interpretation Boundary

These metrics are descriptive and non-rescuing. They cannot alter a root,
mechanics gate, scientific gate, development or confirmation status,
confirmation authorization, claim tier, or any full-cohort estimate.

- Better averaged ranking and error than both draws is consistent with branch
  sampling variance being reduced by averaging.
- Strong draw agreement with poor ranking is consistent with shared model
  bias, support misspecification, or endpoint noise; this audit cannot identify
  which cause is responsible.
- One strong and one weak draw indicates draw sensitivity.

None of these patterns is a causal decomposition. A future model ensemble may
be designed from completed development histories only; confirmation remains
untouched and no model may be selected from confirmation performance.
