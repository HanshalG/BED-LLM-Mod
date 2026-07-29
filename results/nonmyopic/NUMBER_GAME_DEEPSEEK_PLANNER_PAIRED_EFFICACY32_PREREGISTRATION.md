# Number Game DeepSeek Planner Paired Efficacy32 Preregistration

Date frozen: 2026-07-29

## Purpose

Test whether DeepSeek V4 Flash is sufficiently effective to replace Qwen3.7
Plus for future scaled Number Game planning runs. This follows a mechanics-only
exact-10 screen in which DeepSeek passed all linked retained-support gates at
5.36 times lower cost than Qwen.

The experiment is a model-choice gate, not a new headline efficacy cohort.
It pairs the candidate planner with an already frozen external evaluation bank
so model quality is not confounded with target or validation sampling.

## Frozen Inputs

- Candidate planner: `deepseek/deepseek-v4-flash`
- Reasoning: disabled
- Temperature: `0.7`
- Tree seeds: `49000..49031`
- Trees: `32`
- Planning calls per tree: `49`
- Expected candidate provider calls: `1568`
- Support update: retained rejuvenation at both rollout steps
- Candidate roots per tree: `8`
- Canonical endpoint: exactly `33` concepts, equal weight
- Validation supports: the eight Gemini 2.5 Flash supports already stored for
  each matching tree in the Qwen external canonical confirmation
- Tree weighting: equal
- Bootstrap: paired tree bootstrap, `20000` samples, seed `66900`

Bound source artifacts:

- Qwen TREES SHA256:
  `39b79f391ae3b613d15794c9dd6c86ef02eb96907fbaa33591b157b2ac19cc63`
- Canonical TARGETS SHA256:
  `9e788da25b8431f457d044e9f7724bcea77312ca989aaf94b92001a21bf01a44`
- Qwen RESULT SHA256:
  `370e1c2923e56fb6a8344558db0a69bd5f86a8b013da7d9452675df380f7f12b`
- DeepSeek exact-10 RESULT SHA256:
  `5342c72b3eb02e9038f7447dded89e9d726785169e9834335f09da91a03b6b71`

No new target-generation or validation-generation model calls are permitted.
The stored validation and canonical endpoint supports must be used unchanged.

## Mechanics Gates

All must pass:

- exactly `32` complete trees;
- exactly `1568` accepted candidate-planner requests;
- HTTP attempts equal accepted requests plus retries;
- no more than `16` retries;
- zero provider-error retries;
- zero reasoning tokens and zero forced exits;
- all responses parse under the existing strict executable-rule parser;
- every initial support has at least `16` valid unique hypotheses;
- every merged first branch has at least `8` hypotheses;
- every merged second branch has at least `4` hypotheses;
- run cost is at most `$0.50`.

No continuation, semantic repair, model-specific parser, replacement tree,
third proposal draw, threshold relaxation, or alternate seed is allowed.

## Required Intelligence Gate

On the same canonical endpoint and stored validation bank, candidate
path-dependent depth three must beat its own myopic-EIG control by:

- at least `8%` relative mean Brier reduction;
- paired tree-bootstrap Brier-difference upper confidence bound below `0`;
- at least `20/32` tree-level Brier wins.

This is the minimum task intelligence required for future scaled use.

## Qwen Non-Inferiority Gate

For each paired tree, compare DeepSeek depth-three endpoint Brier with the
stored Qwen depth-three endpoint Brier. DeepSeek is non-inferior only if the
upper bound of the paired `95%` tree-bootstrap confidence interval for

`DeepSeek Brier - Qwen Brier`

is below `+0.005`.

The margin is absolute Brier risk and was fixed before candidate efficacy was
observed. Report the mean difference, interval, and win/tie/loss counts.

## Decision Rule

- If mechanics, required intelligence, and Qwen non-inferiority all pass,
  DeepSeek V4 Flash becomes the default planner for future exploratory and
  scaled Number Game runs. Qwen remains a cross-model robustness planner.
- If required intelligence passes but non-inferiority fails, DeepSeek remains
  a lower-cost exploratory frontier point but does not replace Qwen for
  paper-critical confirmations.
- If required intelligence fails, Qwen3.7 Plus remains the default and the
  DeepSeek route closes without a prompt or threshold repair.

The existing Qwen result and the currently running fresh
dynamic-versus-fixed confirmation are unaffected by this model-choice gate.
