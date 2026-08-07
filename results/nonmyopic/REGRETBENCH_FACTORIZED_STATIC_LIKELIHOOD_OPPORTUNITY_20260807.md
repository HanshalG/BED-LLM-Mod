# RegretBench Factorized Static-Likelihood Opportunity

Date: 2026-08-07, before any RegretBench policy response or endpoint.

## Finding

The current SMC transition prompt performs two logically different operations
in one history-conditioned model response:

1. retain or revise the semantic particle population after a simulated
   clarification dialogue; and
2. predict future replies for every revised particle and follow-up question.

The second operation therefore implements a history-updated likelihood of the
form `p_LLM(reply | history, particle, question)`. This can leak the simulated
branch context into the second-step likelihood and make a branch look more
predictable than it will be on a realized trajectory. The dynamic planner uses
these future replies directly, while the refresh-matched myopic control does
not value the second question, so this error can specifically inflate the
predicted horizon advantage and damage predicted-to-realized ranking.

## Literature Evidence

[BED-LLM](https://arxiv.org/abs/2508.21184) separates its history-dependent
filtered belief `p_f(theta | history)` from a static likelihood
`p_LLM(reply | theta, question)`. Its authors argue that when `theta` captures
the information needed to predict a reply, adding the full history is
unnecessary context that can cause calibration shifts. Their published
updated-likelihood ablation is worse than static likelihood in 12 of 15
model/dataset combinations, tied in none; the largest reported gap is 14
success-rate points. The three exceptions are small gains of 1--4 points.

This applies directly to RegretBench: each semantic particle contains an
interpretation and final answer intended to represent the latent user intent.
The particle population should change with dialogue, but a future reply should
be predicted from the resulting particle and question rather than from the
dialogue a second time.

[Xiong (2026)](https://arxiv.org/abs/2605.05851) independently reports an
evaluation--generation gap and systematic Bayesian-like biases in LLM
hypothesis updating. This reinforces separating semantic support generation
from likelihood evaluation rather than trusting a single generative response
to supply both.

## Implemented Core

`scripts/regretbench_static_child_likelihood.py` constructs a separate static
annotation request from:

- the original public ambiguous prompt;
- exactly eight fixed child interpretations and final answers; and
- exactly four fixed child questions.

It excludes:

- the simulated or realized dialogue;
- parent or child probabilities and prior weights;
- retain/revise labels and lineage metadata;
- all predicted replies produced by the transition; and
- hidden truth, CIG slots, aliases, and official mappings.

The strict parser replaces only the eight-by-four reply matrix. It preserves
every child interpretation, final answer, probability, question, lineage
field, and transition diagnostic exactly, and records both source and
factorized support hashes.

This core makes zero model calls and does not modify the unopened Aug 8 or SMC
interfaces.

## Compute-Matched Future Policy

The leading future design is one history-conditioned transition draw followed
by one history-free static-likelihood annotation per branch and arm. It uses
the same two model calls per branch as the current two-transition-draw design:

```text
current:     transition draw 0 + transition draw 1
factorized:  transition draw 0 + static likelihood annotation
```

Both conditioned and history-blind arms must use the same architecture,
model-by-call assignment, seed schedule, task/root common random numbers, and
strict lineage gates. The factorized dynamic policy must retain the same
refresh-matched myopic, fixed-parent, EIG, history-blind, fixed-depth-two, and
random controls. No current confirmation cohort may be reused.

## Authorization Boundary

This audit authorizes no paid call. The current frozen RegretBench result runs
first and is reported unchanged. A factorized policy requires a separate
preregistration, exact-10 schema smoke, measured request-cost envelope, and an
untouched cohort. Completed development histories may be used to diagnose the
mechanism, never to relabel the current result or tune confirmation.
