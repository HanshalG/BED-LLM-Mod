# Number Game GPT-5.6 Luna Paired Efficacy32 Preregistration

Date frozen: 2026-08-05

## Purpose

Test whether GPT-5.6 Luna preserves the task-specific non-myopic Number Game
effect at lower cost than Qwen3.7 Plus. Luna passed the separately frozen
exact-10 interface gate; this experiment evaluates policy efficacy on the
same frozen canonical targets and validation supports used for the prior
paired model comparison.

## Frozen Inputs

- candidate planner: `openai/gpt-5.6-luna`;
- reasoning disabled, temperature `0.7`;
- tree seeds `49000..49031`, exactly 32 paired trees;
- 49 planning histories and 1,568 accepted candidate requests;
- retained rejuvenation at both simulated rollout steps;
- eight candidate roots per tree;
- the hash-bound 33-concept canonical endpoint and eight stored independent
  Gemini validation supports per tree;
- no target-generation or validation-generation provider calls;
- paired tree bootstrap with 20,000 samples and seed `1080600`.

Bound artifacts:

- source TREES SHA256:
  `39b79f391ae3b613d15794c9dd6c86ef02eb96907fbaa33591b157b2ac19cc63`;
- source TARGETS SHA256:
  `9e788da25b8431f457d044e9f7724bcea77312ca989aaf94b92001a21bf01a44`;
- source RESULT SHA256:
  `370e1c2923e56fb6a8344558db0a69bd5f86a8b013da7d9452675df380f7f12b`;
- Luna exact-10 RESULT SHA256:
  `9affa0b48ebc7ae2adfc6867e4b5bf56a47e0dc28c92c8894f94ae1896211496`.

## Mechanics Gates

All must pass:

- exactly 32 complete trees and 1,568 accepted planner requests;
- HTTP attempts equal accepted requests plus retries;
- at most 16 retries and zero provider-error retries;
- zero reasoning tokens and forced exits;
- all 49 response shapes per tree parse under the existing strict parser;
- every initial support has at least 16 valid hypotheses;
- every merged first branch has at least 8 hypotheses;
- every merged second branch has at least 4 hypotheses;
- exactly 32 local target-stub invocations and zero target/validation provider
  calls;
- measured cost at most `$0.85`.

## Efficacy Gates

Against Luna's own myopic-EIG policy, path-dependent depth three must achieve:

- at least 8% relative mean Brier reduction;
- paired tree-bootstrap Brier-difference upper bound below zero;
- at least 20 of 32 tree-level wins.

Against the paired stored Qwen depth-three policy, Luna is noninferior only if
the upper endpoint of the 95% interval for `Luna Brier - Qwen Brier` is below
the frozen absolute margin `+0.005`.

All mechanics, own-myopic efficacy, and Qwen noninferiority gates are required
for a full pass. A pass makes Luna eligible for a separately frozen fresh
model-family replication; it does not alter the unopened Qwen fully fresh
protocol or any prior result.

## Budget

The run is guarded by the account-wide `$5.00` Europe/London daily ledger.
The exact-10 block spent `$0.006836192` across both candidates, leaving
`$4.993163808` at freeze time. This run reserves at most `$0.85` and is
expected to cost about `$0.56` from the observed Luna smoke.
