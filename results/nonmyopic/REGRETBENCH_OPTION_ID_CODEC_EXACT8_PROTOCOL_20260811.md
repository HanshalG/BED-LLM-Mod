# RegretBench Option-ID Codec Exact-8 Protocol

Frozen: 2026-08-11 Europe/London, before any option-ID response or source-value mapping.

Status: **prospective codec serving gate; no paid call is authorized until source audit, producer, independent verifier, dated budget wrapper, bindings, and adversarial tests are committed and pushed**.

## Interface

For each of two frozen codec-calibration tasks, a history-free **proposal** emits:

- exactly eight unique semantic hypotheses, each with an interpretation and concise final factual answer;
- exactly four ranked, distinct, single-facet clarification questions; and
- for each question, option IDs `0`, `1`, and `2` with distinct nonempty natural-language labels plus exact option ID `3` labelled `Other / none of these`.

The proposal emits no prior weight, likelihood, predicted free-text reply, source value, lineage, retention label, or endpoint.

A separate history-free **likelihood evaluator** receives the public prompt and fixed proposal. For each indexed hypothesis it emits a nonnegative prior weight and, for each of four questions, exactly four finite nonnegative option likelihoods aligned to IDs `0..3`. Code normalizes each vector and computes categorical mutual information

```text
H[p(option)] - sum_h p(h) H[p(option | h)].
```

The selected question is the deterministic maximum-MI question supported by a frozen code-only mapper using only public facet IDs and reference-question text. Proposal and evaluator prompts contain neither that action metadata nor source facet values. Only after selection does code load the complete CIG, require the full benchmark mapper to agree on the same facet, and then load that facet's unique values.

Only after selection, code loads the unique values of that selected source facet. Two independently seeded **environment codec** calls receive only:

- public task ID and prompt;
- selected public question and its four fixed options; and
- the unique source values as opaque `value_index` records.

Each codec call returns exactly one valid option ID per value index. It sees no intent description, multiplicity, final answer, answer alias, hidden truth, candidate endpoint, or policy score.

## Exact Eight Accepted Responses

```text
2 proposals
+ 2 likelihood evaluators
+ 2 tasks * 2 independently seeded environment codec calls
= 8 accepted responses
```

Calls occur in three fixed stages: proposals, evaluators, codecs. The two codec calls for a task are adjacent and have identical payload/schema/model/temperature but distinct frozen seeds. The two task pairs may use different strict schemas because their selected facets can have different value counts.

## Model, Seeds, And Budget

- exact `deepseek/deepseek-v4-flash-0731`, nonreasoning;
- temperature `0.7` for proposals/evaluators and `0.0` for codecs;
- strict structured outputs;
- proposal max completion `2400` tokens;
- evaluator max completion `2000` tokens;
- codec max completion `800` tokens;
- proposal seeds `202608540000 + task_index`;
- evaluator seeds `202608541000 + task_index`;
- codec seeds `202608542000 + 10 * task_index + replicate_index`;
- concurrency at most `8`;
- exact eight accepted responses, at most four identical-request infrastructure retries, hence at most 12 attempts;
- accepted-request cap `$0.10`; and
- hard account-wide Europe/London daily cap `$5.00` including unrelated use.

## Gates

All are conjunctive:

### Source, schema, transport, privacy

- exact source/protocol/code bindings and pristine output paths;
- exact 8 accepted responses and 8--12 attempts;
- retry identity and at most four retries;
- zero reasoning tokens and forced exits;
- exact strict schemas, finite values, and all prompt privacy audits;
- proposal contains no priors, likelihoods, predicted free-text replies, source values, lineage, or endpoints;
- evaluator receives no source value, dialogue, observed answer, or endpoint; and
- codec receives no intent description, multiplicity, final answer, answer alias, truth, score, or endpoint.

### Proposal and categorical likelihood

- eight unique hypotheses and four distinct interrogative questions per task;
- every question has exact option IDs `0..3`, three distinct substantive labels, and exact final label `Other / none of these`;
- evaluator particle indexes are an exact permutation;
- every prior and likelihood vector is finite, nonnegative, positive-mass, and normalized by code;
- selected question is supported by the public action mapper, the post-selection full benchmark mapper agrees on the same facet, and mutual information is at least `0.05` nats; and
- its two largest evaluator predictive option masses are each at least `0.10`.

### Environment codec executability

- public action metadata and source values are absent from proposal/evaluator payload hashes; complete CIG/source values are loaded only after question selection;
- selected facet has at least two unique nonempty source values;
- both codec responses are exact value-index permutations with valid option IDs;
- the two independently seeded mappings agree on every value;
- at least two option IDs are used across unique values;
- at least two non-`other` option IDs are used;
- at most one unique source value maps to option `3`;
- every one of the two largest evaluator-predictive options is realized by at least one source value; and
- no source value or option mapping is written to the public result; only counts, hashes, agreement, and option coverage are public.

### Budget

- local accepted-request cost is at most `$0.10`; and
- reconciliation records the maximum of posted account-wide spend since the frozen daily boundary and local accepted-request accounting.

## Authorization Boundary

A verified pass authorizes only writing a separate option-ID proposal-evaluator mechanics preregistration on the two untouched mechanics tasks. It does not authorize mechanics calls, development, confirmation, endpoint labels, model escalation, pooling, or a paper claim. Any failure closes this exact interface and seed; no in-place prompt, option, mapping, threshold, or seed repair is allowed.
