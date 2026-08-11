# RegretBench Proposal-Evaluator V1 Exact-20 Smoke Protocol

Frozen: 2026-08-11 Europe/London, before any proposal-evaluator response, simulated branch reply, or policy endpoint.

Status: **prospective mechanics protocol; no paid call is authorized until source replay, producer, independent verifier, dated budget wrapper, bindings, and adversarial tests are committed and pushed**.

## Question

Can an LLM execute path-dependent semantic belief generation while a separate answer-blind LLM evaluation call supplies priors and likelihood replies that code conditions exactly once? The smoke tests the irreducible belief machinery only. It does not measure policy efficacy.

Use exactly the untouched mechanics positions two and three from the frozen factorized-v2 source manifest. Do not load any development or confirmation task.

## Belief Interface

A **proposal** returns exactly eight unique `(interpretation, final_answer)` hypotheses and four ranked, distinct, single-facet clarification questions. It returns no probability, predicted reply, parent index, lineage, retain/revise label, or endpoint value.

An **evaluator** receives the public ambiguous prompt, one fixed proposed support, and either four root questions or five child questions. For every fixed hypothesis it returns a nonnegative proposal-local prior weight and one concise predicted reply per indexed question. It cannot alter or repeat hypothesis text or questions. It receives no dialogue or dedicated observed/simulated-reply field, truth, intent, source slot, official mapping, old prediction, or endpoint. Because conditioned proposal text is itself data-dependent, it may semantically reflect or quote the branch reply; that is the path-dependent proposal being evaluated, not a second explicit observation input.

Code normalizes evaluator weights, groups predicted replies by deterministic text normalization, computes partition EIG, and conditions with a matched-reply indicator exactly once.

## Exact Twenty Accepted Responses

For each of two tasks:

1. one history-free root proposal;
2. one history-free root evaluator over its four questions;
3. select the highest-EIG mapper-supported root question using only that evaluated support;
4. choose the two highest-prior-mass distinct predicted replies as branch A and B, requiring each branch mass at least `0.10` and each reply to equal a real source value for the selected public facet only as a post-selection mechanics gate;
5. for each branch, issue adjacent same-seed conditioned and answer-free proposals;
6. for each resulting support, issue adjacent same-seed answer-blind evaluators over the answered root question followed by that support's four child questions; and
7. exact-condition both conditioned and answer-free supports on the same branch reply.

The fixed schedule is:

```text
2 root proposals
+ 2 root evaluators
+ 2 tasks * 2 branches * (conditioned proposal + answer-free proposal)
+ 2 tasks * 2 branches * (conditioned evaluator + answer-free evaluator)
= 20 accepted responses
```

The answer-free support is regenerated independently for each branch with the paired branch seed. It is then conditioned by code on that branch reply, so generation compute and observation updating are matched without asking an answer-free model to pretend it revised beliefs.

## Frozen Model And Seeds

- exact model: `deepseek/deepseek-v4-flash-0731`;
- reasoning disabled and excluded;
- temperature `0.7`;
- strict structured outputs;
- proposal maximum completion tokens `2000`;
- evaluator maximum completion tokens `1800`;
- root proposal seeds `202608530000 + task_index`;
- root evaluator seeds `202608531000 + task_index`;
- branch proposal seeds `202608532000 + 10 * task_index + branch_index`, shared within each conditioned/answer-free pair;
- branch evaluator seeds `202608533000 + 10 * task_index + branch_index`, shared within each conditioned/answer-free pair;
- concurrency at most `20`;
- accepted-request cap `$0.20`;
- at most four identical-request infrastructure retries, hence at most 24 HTTP attempts; and
- hard account-wide Europe/London daily cap `$5.00`, including unrelated use.

## Mechanics Gates

Every gate is conjunctive:

### Source, transport, and privacy

- exact source/protocol/code/execution bindings and pristine output paths;
- exactly 20 accepted responses and 20--24 attempts;
- retries, if any, preserve payload, schema, model, seed, and temperature exactly;
- zero reasoning tokens and forced exits;
- every proposal and evaluator parses under the exact strict schema;
- all prompt audits pass;
- conditioned proposals contain only the public prompt plus their one visible question/reply pair;
- answer-free proposals contain an empty dialogue;
- evaluators contain only public task identity/prompt, fixed support text, fixed indexed questions, and structural provenance; and
- no evaluator payload contains a dialogue or dedicated branch-reply field, input probability, lineage, old predicted reply, truth, source intent/slot/mapping, or endpoint. Generated support text is not censored if it reflects the conditioning reply.

### Root executability

- every proposal has eight unique hypotheses and four distinct interrogative questions;
- every evaluator is an exact eight-index permutation with finite nonnegative weights and exactly one nonempty reply per fixed question;
- normalized weights are finite and sum to one;
- the selected root question is the deterministic maximum-EIG question among mapper-supported generated questions;
- selected root EIG is positive;
- its top two normalized reply groups are distinct, each has mass at least `0.10`, and each exactly matches at least one source value for the selected facet; and
- no task answer, truth draw, candidate score, or endpoint influences root-question or branch selection.

### Answer signal and exact updating

For each of four task/branch pairs, let `own_mass` be the pre-update probability assigned to that branch reply by its conditioned support evaluator, `opposite_mass` the probability assigned by the other conditioned branch support, and `blind_mass` the probability assigned by the paired answer-free support.

- every `own_mass` is at least `0.10`;
- at least three of four `own_mass - opposite_mass` values are positive and their mean is at least `0.10`;
- at least three of four `own_mass - blind_mass` values are positive and their mean is at least `0.05`;
- each conditioned and answer-free support has positive mass on its branch reply;
- exact conditioning produces finite normalized posterior weights and posterior predictive probability one for the matched reply partition; and
- no branch reply is appended to an evaluator as a separate observation; the evaluator receives only the fixed proposal and questions.

These gates distinguish answer-conditioned semantic proposal value from same-seed answer-free regeneration noise. They do not reward a self-reported transition label.

### Non-myopic opportunity

- each conditioned and answer-free posterior has at least one mapper-supported informative child question whose facet differs from the root facet;
- each selected child score is finite and positive;
- at least one of four conditioned selected child facets differs from its paired answer-free selected facet; and
- at least one conditioned support differs structurally from both its paired answer-free and opposite-answer support.

### Budget

- locally measured accepted-request cost is at most `$0.20`; and
- reconciliation records the maximum of posted account-wide spend since the frozen daily boundary and locally measured accepted-request cost.

## Authorization Boundary

A verified pass authorizes only writing a separate development-policy preregistration. It does not authorize development calls, endpoint labels, confirmation, model escalation, pooling with prior RegretBench runs, or a paper claim. Any source, transport, schema, privacy, semantic-calibration, exact-update, opportunity, or budget failure closes this exact interface and seed. There is no in-place prompt repair or rerun.
