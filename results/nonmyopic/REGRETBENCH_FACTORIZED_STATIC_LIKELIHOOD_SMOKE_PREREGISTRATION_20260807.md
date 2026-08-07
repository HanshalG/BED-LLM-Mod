# RegretBench Factorized Static-Likelihood Exact-10 Smoke

Date frozen: 2026-08-07, before any RegretBench policy response or endpoint.

Status: zero-call implementation protocol. No paid execution is authorized.

## Purpose

Test whether the compute-matched factorized architecture can preserve strict
history-conditioned semantic-particle transitions while obtaining future reply
likelihoods from a separate history-free annotation call.

This is a mechanics smoke only. It does not score candidate roots, compare
policies, access a development or confirmation endpoint, or inspect efficacy.

## Predecessor

The smoke may eventually run only after the primary RegretBench support smoke
and development artifacts are literal passed results under their frozen
protocol. A future dated wrapper must additionally require their independent
verification, reconciled daily ledger, pristine factorized paths, live catalog
coverage, and an account-wide daily reservation. Those execution requirements
are not implemented or authorized here.

## Exact Calls

Use the first two frozen primary smoke tasks. For each task, use root zero and
the banked hidden-truth mapping only for action-support mechanics.

```text
2 initial parent annotations
+ 2 tasks * (1 conditioned transition + 1 history-blind transition)
+ 2 tasks * (1 conditioned static annotation + 1 blind static annotation)
= 10 calls
```

The conditioned transition receives the root question and exact environment
reply. The history-blind transition receives the same initial support and no
dialogue. Each pair is adjacent and shares a seed. Static annotation calls are
also adjacent and share a task seed; each receives its own fixed child support
through the history-free interface, with old replies, weights, lineage, and
dialogue removed.

Frozen seeds:

- initial annotations: `202608420000 + task`;
- transitions: `202608421000 + task`, shared by conditioned/blind;
- static annotations: `202608422000 + task`, shared by conditioned/blind.

Use `deepseek/deepseek-v4-flash-0731`, temperature `.7`, structured outputs,
at most 2,400 completion tokens, and no reasoning.

## Passage Gates

All must pass:

- exactly ten accepted requests, ten HTTP attempts, and ten parsed responses;
- exactly two initial, four transition, and four static-annotation responses;
- zero retries, provider retries, reasoning tokens, and forced exits;
- exact initial parent-index annotation and no parent regeneration;
- exact transition lineage with eight unique children and two through six
  retained parents;
- static annotations preserve all child particles, probabilities, questions,
  and transition diagnostics while replacing only the eight-by-four reply
  matrix;
- all four static payload audits exclude dialogue, probabilities, lineage,
  old predicted replies, and hidden truth;
- each initial support has at least two informative roots;
- every factorized child support has at least one informative follow-up;
- both first questions and all four selected second questions are supported;
- conditioned second actions are distinct from their first action;
- each exact realized second reply has positive likelihood under its
  factorized child support;
- conditioned/blind transition and annotation adjacency and seed pairing are
  exact; and
- measured smoke cost is at most `$0.20`.

No truth mass, selected-root benefit, policy outcome, or favorable subset may
enter passage.

## Boundary

A smoke pass can authorize only a separate factorized-policy preregistration.
It cannot authorize paid development, alter the current RegretBench or SMC
result, open confirmation, or change any claim tier. Before any paid smoke,
implement and bind a producer-independent raw-response verifier and a dated
transactional budget wrapper.
