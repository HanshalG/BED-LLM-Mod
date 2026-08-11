# RegretBench Factorized V2 Exact-10 Smoke Protocol

Frozen: 2026-08-11 Europe/London, before any factorized-v2 model response,
hidden-intent draw, or policy endpoint.

Status: **prospective mechanics protocol; paid execution remains unauthorized
until the producer, independent verifier, dated budget wrapper, and binding
tests exist**.

## Purpose

Test a new belief interface that separates path-dependent semantic support from
history-free likelihood annotation and then applies the simulated observation
with exact Bayesian code. This is not a rerun or repair of the failed Bongard
visual interface and does not use any prior RegretBench task.

Use the first two task IDs in the fresh mechanics split from
`regretbench_factorized_v2_source_audit/SOURCE_PROTOCOL_MANIFEST.json`.
No candidate root score, selected policy, development endpoint, confirmation
endpoint, or efficacy comparison may be computed.

## Exact Ten Accepted Responses

For each of the two tasks:

1. one history-free root response emits exactly eight semantic particles,
   four clarification questions, and one predicted reply per particle/question;
2. choose root question zero and use the official semantic mapper only to
   construct one mechanics answer;
3. one answer-conditioned and one same-seed history-blind transition response
   retain or revise the eight indexed particles and propose four child
   questions;
4. one answer-conditioned and one same-seed history-blind static annotation
   response receives fixed child particles plus five fixed questions: the
   already answered root question followed by the four child questions.

The schedule is therefore:

```text
2 root supports
+ 2 tasks * (conditioned transition + history-blind transition)
+ 2 tasks * (conditioned static annotation + blind static annotation)
= 10 accepted responses
```

The transition response's own predicted replies are discarded. The static
annotation sees no dialogue, supplied answer, particle weight, lineage label,
old predicted reply, hidden intent, official mapping, or endpoint. It predicts
only how each fixed child particle would answer the five fixed questions.

## Exact Observation Update

For each answer-conditioned branch, match the official simulated answer only
against the static annotations for question zero, the already answered root
question. Require at least one matching child particle. Multiply child prior
weights by the deterministic matched-reply indicator, normalize, and only then
compute EIG over child questions one through four. The history-blind branch is
not conditioned on the withheld answer.

This update makes answer use executable rather than trusting the transition
generator to absorb the dialogue. The child support can still change
path-dependently, so the LLM remains responsible for the semantic transition.

## Model And Seeds

- model: exact `deepseek/deepseek-v4-flash-0731`;
- reasoning: disabled and excluded;
- temperature: `0.7`;
- strict structured outputs;
- maximum completion tokens per request: `2400`;
- root seeds: `202608510000 + task_index`;
- transition seeds: `202608511000 + task_index`, shared by conditioned/blind;
- static seeds: `202608512000 + task_index`, shared by conditioned/blind;
- concurrency: at most `10`;
- accepted-request smoke cap: `$0.20`;
- hard account-wide Europe/London daily cap: `$5.00`.

## Transport Contract

Exactly ten accepted responses are required. Infrastructure retries may repeat
only the identical payload, response schema, model, and seed, with at most four
additional HTTP attempts total. Every retry and accepted-request cost is
recorded. A partial accepted set, changed payload, changed seed, schema error,
unclassified transport error, or fifth retry fails closed and cannot be
resumed. This bounded retry rule is part of the new transport interface and
does not reinterpret the August 8 RegretBench failure.

## Passage Gates

All must pass:

- exact source/protocol/code bindings and pristine output paths;
- exactly ten accepted and parsed responses, with HTTP attempts in `[10,14]`;
- retries are an identical-request subset and remain within four;
- zero reasoning tokens and forced exits;
- root supports contain eight unique semantic particles and four distinct
  interrogative questions, with at least two informative questions per task;
- every transition is an exact eight-parent permutation with two through six
  retained and the rest revised, and conditioned/blind calls are adjacent with
  identical task seeds;
- every static response is an exact eight-child permutation with exactly five
  predicted replies per child;
- static payloads contain exactly the fixed child text and five questions and
  exclude dialogue, answers, probabilities, lineage, old replies, truth, and
  benchmark mappings;
- static parsing preserves every child interpretation, final answer, child
  question, and transition diagnostic while replacing only likelihood replies;
- both official first answers are supported by the mapper and have at least one
  matching child under the conditioned static annotation;
- exact conditioning gives finite normalized weights and posterior predictive
  probability one to the matched reply set;
- every conditioned and blind child has at least one informative future
  question after conditioning rules are applied;
- each selected conditioned second question is mapper-supported, differs in
  semantic facet from the first, and its realized reply has positive static
  likelihood;
- all privacy audits pass; and
- total accepted-response cost is at most `$0.20`.

No truth mass, final-answer accuracy, selected-root benefit, subgroup, or
favorable endpoint may enter passage.

## Authorization Boundary

A verified smoke pass authorizes only a separate, prospectively frozen
factorized-v2 policy protocol on the fresh development split. It does not
authorize development calls by itself, confirmation, a paper claim, or pooling
with prior RegretBench/Bongard results. A mechanics failure closes this exact
interface. A transport failure is banked but cannot be retried in place.
