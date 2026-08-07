# Bongard Luna Contrastive-Prompt Amendment

Date frozen: 2026-08-07, before any Bongard model response.

## Reason

The official Bongard-OpenWorld task defines the hidden concept as a free-form
property depicted by positive examples and not depicted by negative examples.
The frozen prompt asked the VLM to infer a hidden concept and explain the
labelled examples, but did not state that class contrast explicitly.

This matters because the experiment tests whether a new observed label changes
the VLM's generated belief state. A model that treats `positive` and `negative`
as opaque category names may satisfy the JSON schema while failing the intended
concept-induction task. Prior Bongard work also reports that multimodal models
often struggle to use newly supplied information, making task clarity a
validity requirement rather than a performance hint.

## Exact Change

Add this requirement to every semantic-belief request:

> A valid rule describes a visually testable property that is present in
> positive examples and absent from negative examples; use both classes
> contrastively.

The instruction does not reveal an unobserved label, image role, source UID,
concept, caption, filename, dataset position, candidate identity, or endpoint
identity. All images remain under opaque IDs and all unobserved labels remain
unknown.

## Invariants

This amendment changes no task, image, observed history, response schema,
hypothesis count, analytical posterior, EIG formula, policy, control, seed,
request count, budget, endpoint, threshold, split, or execution date. The
development and confirmation manifests must be regenerated before any call so
their implementation bindings cover the exact clarified prompt and this
amendment. The old manifest hashes become superseded without opening any model
response or scientific endpoint.
