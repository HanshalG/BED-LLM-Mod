# Ambig-IaC Regenerated-Support First-Link Smoke Preregistration

Date: 2026-07-25

## Question

Can non-myopic scoring over an LLM's path-dependent regenerated hypothesis
support rank first clarification questions by externally measured final
specification quality better than:

1. immediate one-question EIG; and
2. depth-two EIG on the original fixed particle support?

This is a first-link smoke, not a powered result. A pass authorizes a separately
frozen untouched-task confirmation. A failure closes this exact interface; task,
prompt, parser, model, or threshold changes require a new experiment and cannot
repair this smoke.

## External Environment And Frozen Tasks

- Benchmark: Ambig-IaC.
- Official repository: `https://github.com/agent4ops/ambig-iac`.
- Commit: `4b50c142ed4638a4caaee9ca0b92d8d0e5b8c8cb`.
- Dataset tree SHA-256:
  `1f2d20cadb147962fac104e037d75c53d68d1a9b092050a80513d3751c683f63`.
- Dataset: all 300 released AWS Terraform ambiguity tasks.
- Structural eligibility, computed without model calls: 3-12 target resources,
  2-20 typed dependency edges, and at least three explicit attributes.
- Eligible tasks: 126.
- Selection seed: `24372`.
- Frozen eligible-manifest SHA-256:
  `3df98343ac6d03e15c43d94906db6a780293473c2f4f5f4b719e7e8d712ff34d`.
- Smoke tasks: `272`, `66`, `156`, the first three IDs in the seeded eligible
  order.

The structural audit read aggregate target-plan counts to establish eligibility.
No task was selected using a candidate question, model response, root score, or
realized endpoint. The released target plans are externally authored reference
specifications, not secret benchmark data; the harness nevertheless seals them
from every model call and target-blind score until all 135 responses have frozen.

## LLM-Owned Belief Dynamics

- Model: `openai/gpt-5.4-mini` through OpenRouter.
- Reasoning: disabled (`reasoning_effort: none`); zero reasoning tokens required.
- Temperature: `.7`.
- Output limit: 1,200 tokens.
- Concurrency: 64.
- Particle population: five independently sampled structured specifications per
  state.
- Minimum valid population: four; no repair or scientific retry.
- A particle is strict JSON with exactly `resources`, `topology`, and
  `attributes`.
- Invalid JSON, invalid Terraform addresses, unknown labels, and particles that
  contradict their clarification history are filtered as failed samples.
- Duplicate valid particles are retained because frequency represents generated
  belief mass.

Each initial particle becomes an exact proposition set:

- resource-count threshold features;
- typed resource-dependency features; and
- resource-type/explicit-attribute-key features.

The four root questions are selected target-blind from varying propositions,
with one highest-entropy question from each available dimension before filling
remaining slots by entropy. Every root is simulated under both yes and no. The
LLM then regenerates five fresh structured particles conditioned on the complete
one-answer history. This regeneration, rather than a fixed enumerated support,
is the irreducible LLM belief transition under test.

## Frozen Scores

All scores are in nats and are frozen before target plans load.

- **Myopic:** binary entropy of the root proposition on the initial particles.
- **Fixed depth two:** root entropy plus expected maximum follow-up proposition
  entropy after deterministically filtering the initial particles.
- **Regenerated depth two:** root entropy plus expected maximum follow-up
  proposition entropy on the LLM-regenerated yes/no branch populations. Branch
  probabilities come from the initial particles.

Ties use original root order.

## External Endpoint

After every response and score is frozen, the harness parses the released target
Terraform plan into the same three proposition dimensions. For each candidate
root independently:

1. answer the root exactly from target-feature membership;
2. enter that root's regenerated branch;
3. select its highest-entropy follow-up proposition;
4. answer the follow-up exactly from target-feature membership;
5. filter the branch particles by that answer;
6. select the deterministic posterior medoid; and
7. score macro-F1 across resource, topology, and attribute proposition sets.

Thus every root receives the endpoint under an oracle-best second question on
its own regenerated support. This isolates first-root ranking and avoids adding
a second LLM action-selection link. Posterior mean similarity and empty-support
status are descriptive diagnostics.

## Exact Call And Cost Contract

- Initial calls: `3 * 5 = 15`.
- Branch regeneration calls: `3 * 4 * 2 * 5 = 120`.
- Total physical requests and HTTP attempts: exactly 135.
- Transport retries: zero.
- Forced exits: zero.
- Reasoning tokens: zero.
- Hard run-cost cap: `$1.50`.
- Projected cost: `$0.75`.
- No OatML cluster resources.

The live OpenRouter credit endpoint reported `$44.241497884` remaining before
implementation. The protected reserve is `$25` through Monday 2026-07-27. New
work is additionally capped at `$15` through Monday, anchored at cumulative
project-ledger spend `$86.14330481920742`; this smoke can consume at most `$1.50`
of that allowance. The live balance must be checked again immediately before
the paid command.

## Frozen Pass Gates

The smoke passes only if every gate below passes:

1. exactly 135 physical requests and exactly 135 HTTP attempts;
2. zero transport retries, reasoning tokens, and forced exits;
3. every initial and branch population has at least four valid particles;
4. every task's four roots cover at least two proposition dimensions;
5. at least two tasks have at least `.05` range in realized root endpoints;
6. regenerated depth two selects a different root from myopic on at least two
   tasks;
7. mean within-task Spearman correlation between regenerated depth-two score
   and endpoint is at least `.20`;
8. that mean correlation exceeds the myopic mean by at least `.10`;
9. regenerated selection exceeds myopic selection by at least `.03` mean
   endpoint;
10. regenerated selection exceeds fixed-depth-two selection by at least `.02`
    mean endpoint;
11. no regenerated-selected root has an empty posterior; and
12. reported adapter cost is at most `$1.50`.

Undefined task-level correlations are omitted from their method's mean. If no
finite correlation exists, the corresponding correlation gate fails. No
threshold will be relaxed after responses.

## Zero-Call Verification

Before this preregistration, the complete harness was run with a deterministic
local fixture:

- exact 135 simulated requests and attempts;
- zero retries, reasoning tokens, forced exits, or cost;
- all population, branch-consistency, source-pin, delayed-target, scoring, and
  artifact paths completed;
- strict JSON artifact serialization completed; and
- the fixture failed the efficacy gates, as expected.

Focused tests:

```text
pytest -q tests/test_ambig_iac_first_link_smoke.py
6 passed
```
