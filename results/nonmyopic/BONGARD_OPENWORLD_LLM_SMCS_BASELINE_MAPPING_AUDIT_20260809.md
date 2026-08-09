# Bongard OpenWorld LLM-SMC-S Baseline Mapping Audit

Date: 2026-08-09 (Europe/London)

Status: **exact zero-call implementation audit before any Bongard response**.

## Question

After adding the closest-prior citation, does the frozen Bongard policy family
actually include the relevant operational baseline from Piriyakulkij et al.
(NeurIPS 2024): online LLM belief revision followed by greedy one-step
information-gain experiment selection?

Primary source:
<https://papers.nips.cc/paper_files/paper/2024/file/5f1b350fc0c2affd56f465faa36be343-Paper-Conference.pdf>.

## Frozen Myopic Path

`myopic_width` does the following for every task:

1. It computes endpoint-predictive EIG for every selectable image from the
   initial Luna-generated semantic belief.
2. It selects the deterministic argmax as the first query.
3. It observes that image's realized label.
4. It uses the same answer-conditioned regenerated Luna branch belief available
   to the non-myopic policy.
5. It recomputes endpoint-predictive EIG over the remaining images and selects
   the deterministic argmax as query two.
6. It uses the same terminal belief-generation and endpoint-scoring machinery
   as the other policies.

The existing task-policy function implements steps 1--5 directly. The augmented
first-action invariance regression independently recomputes both EIG vectors and
both argmax selections from the root and realized branch beliefs.

Thus the dynamic-depth-two versus `myopic_width` comparison isolates prospective
lookahead at the first query while preserving online answer-conditioned LLM
belief revision during actual execution. It is not a comparison against a
static-belief greedy policy.

## Relationship To LLM-SMC-S

This is the correct operational analogue, not a reproduction of LLM-SMC-S.

Shared structure:

- natural-language/semantic particles represent uncertainty;
- a real observation triggers an LLM-mediated belief update;
- the next experiment is chosen greedily by one-step information gain.

Material differences:

- Bongard queries come from a fixed sealed image set, whereas LLM-SMC-S can use
  an LLM to propose experiments;
- Bongard regenerates a structured predictive belief, whereas LLM-SMC-S uses a
  selective proposal kernel, importance weights, and resampling;
- Bongard optimizes predictive information about sealed endpoint labels,
  whereas LLM-SMC-S measures information over its current natural-language rule
  particles.

The paper may therefore say that `myopic_width` is an online-regeneration greedy
baseline matching the closest prior's design pattern. It must not call the
baseline LLM-SMC-S, claim an empirical comparison with that implementation, or
claim novelty for online LLM hypothesis revision plus information gain.

## Scope

This audit changes no production code, model, prompt, task, seed, action,
likelihood, policy, endpoint, threshold, gate, request count, cost cap,
authorization, or paper outcome mapping. It opens no labels or endpoints, makes
zero model calls, and costs `$0`.
