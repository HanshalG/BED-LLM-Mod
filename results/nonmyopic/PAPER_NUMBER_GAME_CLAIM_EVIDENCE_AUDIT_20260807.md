# Paper Number Game Claim-Evidence Audit

Date: 2026-08-07

## Scope

This zero-call audit updates the manuscript after the mandatory fully fresh
history-blind control. It does not reopen the sealed endpoint or change any
source, control, or pooled-analysis status.

## New Evidence Boundary

The control completed exactly 3,072 Qwen responses with no retries, provider
errors, reasoning tokens, forced exits, or strict-JSON failures. One of 1,536
branch slots nevertheless missed the frozen valid-draw/support floors. The run
therefore stopped at `mechanics_failed`, and `endpoint_accessed` remained
`false`.

The manuscript may no longer describe this control as pending. It may state
that the fully fresh causal policy--mechanism replication remains
unestablished because its mandatory control failed before endpoint scoring.
It must not infer a policy contrast from the unopened endpoint.

The applicable mechanism evidence remains the two disjoint matched-prompt
controls and their retrospective cohort-stratified synthesis: second-stage MSE
and coverage improve, and root-specific conditioning benefit predicts realized
advantage, while the pooled selected-root mean remains null.

## Verification

The new claim-manifest bundle binds the public failure artifact, exact request
and transport counts, mechanics failures, cost, and sealed endpoint. All 38
claim bundles pass hash and value validation. The focused validator tests pass
12/12, and the draft compiles to six pages with all 14 required limitation
topics and both required figures.
