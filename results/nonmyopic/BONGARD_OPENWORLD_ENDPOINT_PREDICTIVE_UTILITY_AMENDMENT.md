# Bongard OpenWorld Endpoint-Predictive Utility Amendment

Date frozen: 2026-08-08, before any Bongard mechanics, development, or
confirmation model response and before any candidate or endpoint label was
accessed.

This prospective amendment changes only the deterministic utility used to
select queries from already-generated semantic beliefs. It does not change the
task split, visible images, model, prompts, request tree, request seeds, query
budget, terminal regeneration, daily cost caps, scientific endpoints, or
success thresholds.

## Motivation

The earlier protocol scored entropy reduction over ten free-form rule strings.
Those strings have no canonical identity. A query may separate paraphrased or
otherwise endpoint-equivalent rules while leaving every held-out prediction
unchanged. For example, with four equally weighted particles, a query with
likelihoods `(0.01, 0.99, 0.01, 0.99)` can almost perfectly separate particle
identities while preserving a 0.5 predictive probability for an endpoint whose
particle probabilities are `(0.1, 0.1, 0.9, 0.9)`. A second query with
likelihoods `(0.25, 0.25, 0.75, 0.75)` has lower particle EIG but positive
information about that endpoint. The old utility selects the first query even
though it cannot improve either registered endpoint prediction.

The old particle EIG remains logged as a diagnostic. It no longer selects an
action.

## Frozen Utility

The planner knows the two opaque endpoint image IDs, whose pixels were already
included in every belief request, but never sees their labels. This is
transductive, target-aware Bayesian experimental design; it is not endpoint
label access. Candidate and endpoint roles remain hidden from the VLM prompt.

For belief `B`, weights `w`, and endpoint IDs `E`, define the target risk in
nats as the sum of marginal Bernoulli entropies:

```text
H_E(B,w) = sum_{e in E} h(sum_i w_i p_i(y_e=1)).
```

This is the Bayes risk under the registered marginal endpoint log loss. The
primary Brier endpoint remains unchanged and is evaluated only after all query
selection is complete.

The one-step score for query `a` is:

```text
PIG(a;B,w) = H_E(B,w)
  - sum_{y in {0,1}} P_B(y|a,w) H_E(B, update(B,w,a,y)).
```

`myopic_width` maximizes this score. Every nonrandom policy's realized second
query also maximizes this score on its prescribed support.

For fixed-support depth two, analytically update the root support after both
simulated answers and choose the second query with minimum expected terminal
target entropy. For answer-conditioned dynamic support:

```text
dynamic_score(a) = H_E(root)
  - sum_y P_root(y|a)
      min_{b != a} E_{z ~ branch[a,y]} H_E(update(branch[a,y],b,z)).
```

This score directly values whether the first answer induces a regenerated VLM
belief that supports useful endpoint prediction and a useful second query. It
may be negative when support regeneration increases expected endpoint
uncertainty.

The history-blind score uses the same formula, but its paired regenerated
support has not seen the simulated first answer; the evaluator analytically
applies that answer before scoring the second query. The matched fixed-score
control selects its first query with fixed-support predictive depth two and its
second query with the same answer-conditioned predictive rule as the dynamic
policy.

The shuffled control decomposes each signed dynamic score into its myopic
predictive score plus a signed continuation value, then applies the already
frozen derangement to the complete continuation values. Negative continuation
values are valid and retained.

## Integrity And Interpretation

- Endpoint labels remain absent during root, branch, action, and second-query
  selection.
- All policies use the same existing semantic particles and image
  probabilities supplied by the VLM. No classical rule bank or external
  likelihood model is introduced.
- No additional model request is made. The maximum accepted responses, HTTP
  attempts, token exposure, and dollar exposure are byte-for-byte unchanged.
- Ranking fidelity continues to compare each prospective first-action score to
  realized endpoint Brier over all first-action terminal histories. It now
  diagnoses support-regeneration fidelity under an endpoint-aligned utility.
- All development and confirmation task identities, paired comparisons,
  bootstrap procedures, change-count gates, and effect thresholds remain
  fixed.

Mechanics, development, naive-manifest, August 10 wrapper, and confirmation
interfaces and manifests must advance and bind this amendment before any paid
Bongard request. Earlier unopened executable manifests are invalidated; their
historical files remain preserved.
