# RegretBench SMC Transition-Model Ensemble Audit

Date: 2026-08-07, before any RegretBench SMC response.

## Decision

Keep both frozen SMC transition draws on
`deepseek/deepseek-v4-flash-0731`. Do not replace one draw with GPT-5.6 Luna
in the unopened development or confirmation interfaces.

This is a zero-call design audit. It changes no model, prompt, seed, draw,
threshold, budget, endpoint, or execution binding.

## Question

The primary-claim power audit identifies predicted-to-realized ranking fidelity
on changed roots as the likely bottleneck. A tempting repair is a cost-matched
transition ensemble: DeepSeek for draw zero and Luna for draw one, while using
the same model-by-draw assignment and seed in every root and in the conditioned
and history-blind arms.

That design would preserve within-draw pairing, but it is not currently
evidence-based. The banked comparisons measure schema reliability and support
generation in Number Game, not RegretBench transition-risk residuals. They do
not estimate either model's RegretBench ranking error, the covariance of those
errors, or the ranking fidelity of their average.

## Direct Evidence

All numbers below come from already banked nonreasoning requests.

| Model and instrument | Requests | Prompt/request | Completion/request | Cost/request | Strict or semantic result |
|---|---:|---:|---:|---:|---|
| Luna exact-10 | 10 | 286.6 | 548.5 | $0.000357760 | passed |
| DeepSeek 0731 exact-10 | 10 | 248.7 | 1,072.0 | $0.000325859 | gated null; one zero-valid conditioned draw |
| Luna reliability-128 | 128 | 298.8 | 697.4 | $0.000448305 | gated null; 4 forced exits and strict failures |
| DeepSeek 0731 reliability-128 | 128 | 268.8 | 874.2 | $0.000269357 | gated null; 128/128 strict parses, low-support tail |

Replacing 4,096 DeepSeek transition calls with Luna would add about `$0.13`
under the exact-10 observed request costs or `$0.73` under the reliability-128
observed request costs. The latter would move the current `$3.10` projected
development spend to roughly `$3.83`, above the frozen `$3.50` stage cap. This
is an extrapolation, not a RegretBench cost measurement; the structured output
shape differs. It nevertheless shows that the ensemble is not safely
cost-matched under the existing budget.

## Statistical Reason

For two transition-risk estimates with equal marginal variance `v` and error
correlation `rho`, averaging gives variance `v(1 + rho) / 2`. Cross-model draws
help only if their reduction in correlated error outweighs any difference in
bias and serving reliability. Diversity by itself is insufficient: a second
model can lower variance while shifting all candidate risks in the wrong
direction, or can introduce model identity as a draw effect.

No banked artifact estimates `rho` or relative RegretBench bias. Changing the
model now would therefore optimize the design using an unmeasured efficacy
surrogate. It would also require a new prompt/schema smoke, cost envelope,
producer and verifier routing, confirmation binding, and paper protocol.

## Preserved Execution

- Draws zero and one remain independent DeepSeek generations.
- The same seed remains shared across roots for a fixed task, particle, and
  draw.
- Each conditioned/history-blind pair remains adjacent and seed matched.
- The first development result remains a clean test of the frozen LLM-native
  transition model rather than an outcome-adaptive model search.
- A ranking-only primary null will be interpreted as simulator-fidelity
  failure; a Brier-effect null will be interpreted as a substantive nonmyopic
  null, as registered in the power audit.

## Future Ensemble Gate

A later interface may test a mixed nonreasoning ensemble only after the current
interface closes. It must use completed, non-confirmation histories to estimate
per-model candidate-risk residuals and their cross-model correlation, freeze a
model-by-draw routing schedule, prove the full request envelope fits a new daily
cap, and validate both models on the exact RegretBench transition schema.
Neither support-count superiority nor generic benchmark score is sufficient.
