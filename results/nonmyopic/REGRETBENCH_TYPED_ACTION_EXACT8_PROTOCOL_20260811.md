# RegretBench Typed-Action Exact-8 Protocol

Frozen: 2026-08-11 Europe/London, before any typed-action response.

Status: **prospective calibration gate; no paid call is authorized until the source audit, producer, independent verifier, dated budget wrapper, bindings, and adversarial tests are committed and pushed**.

## Interface

For each of two frozen calibration tasks, a history-free proposal receives the public ambiguous prompt and exactly four code-rendered action records. It returns:

- exactly eight unique semantic hypotheses, each with an interpretation and concise final factual answer; and
- exactly four typed action records, one for every allowed `action_id`, each containing option IDs `0`, `1`, and `2` with distinct nonempty semantic labels plus exact option ID `3` labelled `Other / none of these`.

The model does not emit question text. Code joins each returned `action_id` to the frozen public rendered question. The proposal emits no prior, likelihood, free-text predicted reply, source value, lineage, retention label, or endpoint.

A separate history-free evaluator receives the public prompt, allowed actions, and fixed proposal. For each indexed hypothesis it emits a finite nonnegative prior weight and, for every action ID, four finite nonnegative option likelihoods aligned to IDs `0..3`. Code normalizes each vector and computes categorical mutual information

```text
H[p(option)] - sum_h p(h) H[p(option | h)].
```

Code selects the deterministic maximum-MI action, breaking ties by public action order. Only then does code load the complete CIG, require the action ID and rendered question to replay exactly, and load unique source values for the selected facet.

Two independently seeded environment codec calls receive only the selected public action/question, its four fixed option labels, and indexed unique source values. Each returns exactly one valid option ID per value index. They see no intent description, multiplicity, prior, final answer, alias, truth, endpoint, or policy score.

## Exact Eight Accepted Responses

```text
2 proposals
+ 2 categorical likelihood evaluators
+ 2 tasks * 2 independently seeded environment codec calls
= 8 accepted responses
```

Calls occur in three fixed stages. Codec calls are adjacent task-specific pairs and may use different strict schemas because selected facets can have different value counts.

## Model, Seeds, And Budget

- exact `deepseek/deepseek-v4-flash-0731`, nonreasoning;
- proposal/evaluator temperature `0.7`; codec temperature `0.0`;
- strict structured outputs;
- proposal/evaluator/codec maximum completions `2400` / `2000` / `800` tokens;
- proposal seeds `202608550000 + task_index`;
- evaluator seeds `202608551000 + task_index`;
- codec seeds `202608552000 + 10 * task_index + replicate_index`;
- concurrency at most `8`;
- exact eight accepted responses and at most four identical-request infrastructure retries;
- accepted-request cap `$0.10`; and
- hard account-wide Europe/London daily cap `$5.00`, including unrelated use.

## Gates

All gates are conjunctive.

### Source, Schema, Transport, Privacy

- exact source/protocol/code bindings and pristine output paths;
- exact eight accepted responses and 8--12 HTTP attempts;
- retry identity, at most four retries, zero reasoning tokens, and zero forced exits;
- strict schemas and finite numeric values;
- proposal/evaluator payloads contain public action IDs/questions but no source values, source intent/answer data, observation history, or endpoints;
- source values load only after four root calls and action selection; and
- public outputs contain no source value or private mapping.

### Typed Proposal And Categorical Likelihood

- eight unique hypotheses;
- returned action IDs are an exact permutation of the four allowed IDs;
- every action has ordered option IDs `0..3`, three distinct substantive labels, and exact final other label;
- evaluator particle indexes are an exact permutation;
- every prior and likelihood vector is finite, nonnegative, positive-mass, and normalized by code;
- selected mutual information is at least `0.05` nats; and
- each of the two largest evaluator-predictive option masses is at least `0.10`.

### Environment Codec Executability

- selected full CIG action ID and canonical rendered question replay exactly;
- selected facet has at least two unique nonempty values;
- both codec mappings are complete valid permutations and agree exactly;
- at least two option IDs and at least two non-other IDs are used;
- at most one unique value maps to option `3`; and
- both top evaluator-predictive options are realized by source values.

## Authorization Boundary

A verified pass authorizes only writing a separate typed-action mechanics preregistration on the two untouched mechanics tasks. It does not authorize mechanics calls, development, confirmation, endpoint labels, model escalation, or a paper claim. Any failure closes this exact interface and seed; no in-place schema, prompt, threshold, task, or seed repair is allowed.
